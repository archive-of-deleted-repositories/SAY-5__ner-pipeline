"""Run TransformersExtractor over the CoNLL-2003 test split and report F1.

Usage::

    python scripts/eval_conll.py                     # full test split
    python scripts/eval_conll.py --limit 100         # quick smoke
    python scripts/eval_conll.py --output eval/results/run.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from ner_pipeline.eval import evaluate
from ner_pipeline.extractor import TransformersExtractor


def load_conll_test(
    split: str = "test", limit: int | None = None
) -> tuple[list[list[str]], list[list[int]]]:
    """Load the CoNLL-2003 split via the ``datasets`` library.

    Tries the modern dataset id first, then falls back to the legacy one.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    last_err: Exception | None = None
    for dataset_id in ("eriktks/conll2003", "conll2003"):
        try:
            ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
            break
        except Exception as exc:
            last_err = exc
            ds = None
            continue
    if ds is None:
        raise RuntimeError(f"could not load CoNLL-2003 split '{split}': {last_err}")

    tokens: list[list[str]] = []
    tags: list[list[int]] = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        tokens.append(list(row["tokens"]))
        tags.append(list(row["ner_tags"]))
    return tokens, tags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--model",
        default=os.getenv("NER_MODEL_NAME", "dslim/distilbert-NER"),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSON metrics to this path.",
    )
    args = parser.parse_args()

    print(f"loading CoNLL-2003 split={args.split} limit={args.limit}", flush=True)
    sentences, gold = load_conll_test(args.split, args.limit)
    print(f"  loaded {len(sentences)} sentences", flush=True)

    print(f"loading model {args.model}", flush=True)
    extractor = TransformersExtractor(model_name=args.model)

    print("running extractor...", flush=True)
    result = evaluate(extractor, sentences, gold)

    print()
    print("==== CoNLL-2003 results ====")
    print(f"  precision : {result.precision:.4f}")
    print(f"  recall    : {result.recall:.4f}")
    print(f"  f1        : {result.f1:.4f}")
    print(f"  support   : {result.support}")
    print()
    print(f"  {'type':<6} {'P':>8} {'R':>8} {'F1':>8} {'support':>8}")
    for etype in ("PER", "ORG", "LOC", "MISC"):
        row = result.per_type.get(etype)
        if row is None:
            continue
        print(
            f"  {etype:<6} {row['precision']:>8.4f} {row['recall']:>8.4f} "
            f"{row['f1']:>8.4f} {int(row['support']):>8d}"
        )
    print()

    out_path = args.output
    if out_path is None:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = f"eval/results/{ts}.json"

    payload = {
        "model": args.model,
        "dataset": "conll2003",
        "split": args.split,
        "limit": args.limit,
        "sentences_evaluated": len(sentences),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "metrics": result.to_json(),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
