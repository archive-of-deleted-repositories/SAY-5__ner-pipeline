"""Unit tests for the ingest bench harness."""

from __future__ import annotations

import json
from pathlib import Path

from bench.ingest_bench import (
    BenchResult,
    _build_corpus,
    _build_doc,
    _percentile,
    run_bench,
    write_result,
)
from faker import Faker


def test_build_doc_offsets_match_text() -> None:
    """Each Mention.char_start/end must slice back to its surface_form."""
    Faker.seed(7)
    sample = _build_doc(Faker())
    for m in sample.mentions:
        assert sample.text[m.char_start : m.char_end] == m.surface_form


def test_build_corpus_is_deterministic() -> None:
    a = _build_corpus(50, seed=123)
    b = _build_corpus(50, seed=123)
    assert [s.text for s in a] == [s.text for s in b]


def test_percentile_basic() -> None:
    assert _percentile([], 0.5) == 0.0
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
    # P99 of 100 ascending values is 99.something
    samples = [float(i) for i in range(1, 101)]
    p99 = _percentile(samples, 0.99)
    assert 99.0 <= p99 <= 100.0


def test_run_bench_smoke() -> None:
    """100-doc bench must complete and report sensible numbers."""
    result = run_bench(n_docs=100, seed=0)
    assert isinstance(result, BenchResult)
    assert result.n_docs == 100
    assert result.docs_per_sec > 0
    assert result.mentions_total > 0
    # Each doc has at most ~5 named entities; so total mentions is bounded.
    assert result.mentions_total <= 100 * 5
    assert result.latency_ms_p50 > 0
    assert result.latency_ms_p95 >= result.latency_ms_p50
    assert result.latency_ms_p99 >= result.latency_ms_p95
    # Pipeline must not regress to N+1 on the synthetic corpus.
    assert result.has_n_plus_one is False
    assert result.peak_rss_mb > 0
    assert result.db_query_count_per_doc_mean > 0


def test_write_result_roundtrips(tmp_path: Path) -> None:
    result = run_bench(n_docs=20, seed=1)
    path = write_result(result, tmp_path)
    assert path.exists()
    with path.open() as f:
        payload = json.load(f)
    assert payload["n_docs"] == 20
    assert payload["seed"] == 1
    assert "docs_per_sec" in payload
