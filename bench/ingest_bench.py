"""Bench harness for the ingest pipeline.

Synthesises N short news-style documents (deterministic Faker), times the
full pipeline (text -> extractor -> mention dedup -> write), and emits a
JSON report with docs/sec, latency percentiles, peak RSS, and DB query
counts. Detects N+1 query patterns by asserting query-count-per-doc stays
bounded.

Default backend is in-memory SQLite so the bench is hermetic in CI; pass
``--db-url postgresql+psycopg://...`` to bench against Postgres.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from faker import Faker
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from ner_pipeline.extractor import Mention
from ner_pipeline.models import Base
from ner_pipeline.pipeline import ingest

# News-style templates with predictable entity slots so mention counts are
# realistic without needing the real model loaded.
_TEMPLATES: tuple[str, ...] = (
    "{person} announced a new partnership between {org} and {org2} from offices in {city}, {country}.",
    "Authorities in {city} confirmed that {person} will lead {org}'s expansion across {country}.",
    "{org} reported quarterly results today; {person}, the chief executive, addressed analysts in {city}.",
    "Officials in {country} met with {person} to discuss {org}'s investment plans for {city}.",
    "Reporters in {city} caught up with {person} after the {org} board meeting concluded.",
)


@dataclass(frozen=True)
class DocSample:
    text: str
    mentions: tuple[Mention, ...]


def _build_doc(faker: Faker) -> DocSample:
    """Build one synthetic doc with known char-spans for each entity."""
    template = random.choice(_TEMPLATES)
    fields = {
        "person": faker.name(),
        "org": faker.company(),
        "org2": faker.company(),
        "city": faker.city(),
        "country": faker.country(),
    }
    text = template.format(**fields)

    # Re-locate each filled value to compute exact char offsets.
    type_map = {"person": "PER", "org": "ORG", "org2": "ORG", "city": "LOC", "country": "LOC"}
    mentions: list[Mention] = []
    for slot, value in fields.items():
        if not value:
            continue
        idx = text.find(value)
        if idx < 0:
            continue
        mentions.append(
            Mention(
                surface_form=value,
                char_start=idx,
                char_end=idx + len(value),
                entity_type=type_map[slot],  # type: ignore[arg-type]
                confidence=0.95,
            )
        )
    return DocSample(text=text, mentions=tuple(mentions))


def _build_corpus(n: int, seed: int) -> list[DocSample]:
    random.seed(seed)
    faker = Faker()
    Faker.seed(seed)
    return [_build_doc(faker) for _ in range(n)]


class _ScriptedExtractor:
    """Returns pre-computed mentions per text. Removes model latency from the bench."""

    def __init__(self, samples: list[DocSample]) -> None:
        self._table = {s.text: list(s.mentions) for s in samples}

    def extract(self, text: str) -> list[Mention]:
        return list(self._table.get(text, []))


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS ru_maxrss is in bytes; on Linux it is in kilobytes.
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _percentile(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


@dataclass
class BenchResult:
    n_docs: int
    seed: int
    db_url: str
    docs_per_sec: float
    total_seconds: float
    mentions_total: int
    mentions_per_doc_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    peak_rss_mb: float
    db_query_count_total: int
    db_query_count_per_doc_mean: float
    db_query_count_per_doc_max: int
    has_n_plus_one: bool
    timestamp: str


def run_bench(
    n_docs: int = 10000,
    seed: int = 42,
    db_url: str = "sqlite:///:memory:",
    queries_per_mention_budget: float = 4.0,
) -> BenchResult:
    """Run the bench and return a typed result.

    ``queries_per_mention_budget`` is the upper bound on (queries / mentions)
    per doc we accept. A regression to true N+1 (e.g. one query per mention
    per persisted child) inflates this ratio sharply, so the assertion catches
    it without depending on absolute query counts.
    """
    samples = _build_corpus(n_docs, seed)
    extractor = _ScriptedExtractor(samples)

    engine: Engine = create_engine(db_url, future=True)
    Base.metadata.create_all(engine)

    # Count every emitted SQL statement on this engine so we can detect N+1.
    query_count: dict[str, int] = {"n": 0}

    @event.listens_for(engine, "before_cursor_execute")
    def _count(  # type: ignore[no-untyped-def]  # SQLA event signature
        conn, cursor, statement, parameters, context, executemany
    ) -> None:
        query_count["n"] += 1

    sm = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    latencies_ms: list[float] = []
    per_doc_queries: list[int] = []
    mentions_total = 0

    gc.collect()

    t0 = time.perf_counter()
    with sm() as session:
        for sample in samples:
            qbefore = query_count["n"]
            tic = time.perf_counter()
            res = ingest(
                source="bench",
                text=sample.text,
                extractor=extractor,
                session=session,
            )
            session.flush()
            toc = time.perf_counter()
            latencies_ms.append((toc - tic) * 1000.0)
            per_doc_queries.append(query_count["n"] - qbefore)
            mentions_total += res.entity_count
        session.commit()
    total = time.perf_counter() - t0

    engine.dispose()

    max_qpd = max(per_doc_queries) if per_doc_queries else 0
    mean_qpd = statistics.mean(per_doc_queries) if per_doc_queries else 0.0
    mentions_per_doc_mean = (mentions_total / n_docs) if n_docs else 0.0
    qpm = (mean_qpd / mentions_per_doc_mean) if mentions_per_doc_mean > 0 else 0.0
    has_n_plus_one = qpm > queries_per_mention_budget
    return BenchResult(
        n_docs=n_docs,
        seed=seed,
        db_url=db_url,
        docs_per_sec=(n_docs / total) if total > 0 else 0.0,
        total_seconds=total,
        mentions_total=mentions_total,
        mentions_per_doc_mean=mentions_per_doc_mean,
        latency_ms_p50=_percentile(latencies_ms, 0.50),
        latency_ms_p95=_percentile(latencies_ms, 0.95),
        latency_ms_p99=_percentile(latencies_ms, 0.99),
        peak_rss_mb=_peak_rss_mb(),
        db_query_count_total=query_count["n"],
        db_query_count_per_doc_mean=mean_qpd,
        db_query_count_per_doc_max=max_qpd,
        has_n_plus_one=has_n_plus_one,
        timestamp=datetime.now(UTC).isoformat(),
    )


def write_result(result: BenchResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.timestamp.replace(":", "").replace("-", "").split(".")[0]
    path = out_dir / f"{stamp}.json"
    with path.open("w") as f:
        json.dump(asdict(result), f, indent=2, sort_keys=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-docs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db-url", default="sqlite:///:memory:")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--fail-on-n-plus-one",
        action="store_true",
        help="Exit non-zero if N+1 query pattern is detected.",
    )
    parser.add_argument(
        "--regress-baseline",
        type=Path,
        default=None,
        help="Path to a previous bench JSON to compare against (regression gate).",
    )
    parser.add_argument(
        "--regress-tolerance",
        type=float,
        default=0.30,
        help="Allowed slowdown ratio vs baseline docs_per_sec (default 30%%).",
    )
    args = parser.parse_args(argv)

    result = run_bench(n_docs=args.n_docs, seed=args.seed, db_url=args.db_url)
    path = write_result(result, args.out_dir)

    summary: dict[str, Any] = asdict(result)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nresult written to {path}")

    rc = 0
    if args.fail_on_n_plus_one and result.has_n_plus_one:
        print(
            f"\nFAIL: per-doc query count max={result.db_query_count_per_doc_max} "
            f"(n+1 regression)",
            file=sys.stderr,
        )
        rc = 2

    if args.regress_baseline is not None:
        with args.regress_baseline.open() as f:
            baseline = json.load(f)
        baseline_dps = float(baseline.get("docs_per_sec", 0.0))
        if baseline_dps > 0 and result.docs_per_sec < baseline_dps * (1 - args.regress_tolerance):
            print(
                f"\nFAIL: docs_per_sec={result.docs_per_sec:.1f} dropped >"
                f" {args.regress_tolerance*100:.0f}% vs baseline {baseline_dps:.1f}",
                file=sys.stderr,
            )
            rc = 3

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
