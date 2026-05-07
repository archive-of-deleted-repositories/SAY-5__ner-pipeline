"""Smoke eval: run the real model on 100 CoNLL-2003 test sentences and gate F1."""

from __future__ import annotations

import pytest

from ner_pipeline.eval import evaluate
from ner_pipeline.extractor import TransformersExtractor

pytestmark = pytest.mark.eval_smoke

# Lower bound for distilbert-NER on a 100-sentence subset of CoNLL-2003 test.
# A drop below this is a real regression (or a model swap) worth catching.
F1_FLOOR = 0.85


def test_distilbert_ner_smoke() -> None:
    from scripts.eval_conll import load_conll_test

    sentences, gold = load_conll_test("test", limit=100)
    assert len(sentences) == 100

    extractor = TransformersExtractor(model_name="dslim/distilbert-NER")
    result = evaluate(extractor, sentences, gold)

    print(
        f"\nsmoke F1={result.f1:.4f} P={result.precision:.4f} R={result.recall:.4f} "
        f"support={result.support}"
    )
    assert result.f1 >= F1_FLOOR, f"F1 {result.f1:.4f} below floor {F1_FLOOR}"
