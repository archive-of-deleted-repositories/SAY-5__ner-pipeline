"""Unit tests for the ONNX extractor's pure-python helpers.

The full ``OnnxExtractor.extract()`` path requires a real ONNX model on
disk and HF tokenizer weights, so it lives in the ``onnx-parity-smoke``
CI job. These tests exercise the BIO-grouping and softmax helpers directly
to keep the unit suite hermetic.
"""

from __future__ import annotations

import math

from ner_pipeline.onnx_extractor import _bio_to_mentions, _softmax


def test_softmax_rows_sum_to_one() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 9)).astype("float32")
    out = _softmax(logits, axis=-1)
    assert out.shape == logits.shape
    sums = out.sum(axis=-1)
    assert all(abs(s - 1.0) < 1e-5 for s in sums.tolist())


def test_softmax_argmax_matches_input() -> None:
    import numpy as np

    logits = np.array([[1.0, 5.0, 2.0]], dtype="float32")
    probs = _softmax(logits, axis=-1)
    assert int(probs.argmax(axis=-1)[0]) == 1
    assert math.isclose(float(probs.sum()), 1.0, abs_tol=1e-5)


def _id2label() -> dict[int, str]:
    return {
        0: "O",
        1: "B-PER",
        2: "I-PER",
        3: "B-ORG",
        4: "I-ORG",
        5: "B-LOC",
        6: "I-LOC",
        7: "B-MISC",
        8: "I-MISC",
    }


def test_bio_to_mentions_groups_subwords_into_one_mention() -> None:
    """Three subwords J/AP/AN -> single mention "JAPAN" via word_ids merging."""
    text = "JAPAN beat CHINA"
    # Tokens: [CLS] J AP AN beat CH INA [SEP]
    offsets = [(0, 0), (0, 1), (1, 3), (3, 5), (6, 10), (11, 13), (13, 16), (0, 0)]
    word_ids = [None, 0, 0, 0, 1, 2, 2, None]
    pred_ids = [0, 1, 1, 1, 0, 3, 3, 0]
    pred_scores = [0.0, 0.99, 0.99, 0.99, 0.0, 0.95, 0.95, 0.0]

    mentions = _bio_to_mentions(
        text=text,
        offsets=offsets,
        word_ids=word_ids,
        pred_ids=pred_ids,
        pred_scores=pred_scores,
        id2label=_id2label(),
        min_confidence=0.5,
    )
    assert [(m.surface_form, m.entity_type) for m in mentions] == [
        ("JAPAN", "PER"),
        ("CHINA", "ORG"),
    ]


def test_bio_to_mentions_drops_below_min_confidence() -> None:
    text = "Apple"
    offsets = [(0, 0), (0, 5), (0, 0)]
    word_ids = [None, 0, None]
    pred_ids = [0, 3, 0]  # B-ORG
    pred_scores = [0.0, 0.40, 0.0]

    mentions = _bio_to_mentions(
        text=text,
        offsets=offsets,
        word_ids=word_ids,
        pred_ids=pred_ids,
        pred_scores=pred_scores,
        id2label=_id2label(),
        min_confidence=0.5,
    )
    assert mentions == []


def test_bio_to_mentions_consecutive_same_type_words_merge() -> None:
    """B-PER + I-PER on adjacent words form a single mention."""
    text = "Angela Merkel went home"
    offsets = [(0, 0), (0, 6), (7, 13), (14, 18), (19, 23), (0, 0)]
    word_ids = [None, 0, 1, 2, 3, None]
    pred_ids = [0, 1, 2, 0, 0, 0]  # B-PER, I-PER, O, O
    pred_scores = [0.0, 0.99, 0.98, 0.0, 0.0, 0.0]

    mentions = _bio_to_mentions(
        text=text,
        offsets=offsets,
        word_ids=word_ids,
        pred_ids=pred_ids,
        pred_scores=pred_scores,
        id2label=_id2label(),
        min_confidence=0.5,
    )
    assert len(mentions) == 1
    m = mentions[0]
    assert m.surface_form == "Angela Merkel"
    assert m.entity_type == "PER"
    assert m.char_start == 0
    assert m.char_end == 13


def test_bio_to_mentions_unknown_label_dropped() -> None:
    """Labels mapping to ``None`` (unknown HF group) flush the current span."""
    text = "Foo Bar"
    offsets = [(0, 3), (4, 7)]
    word_ids = [0, 1]
    pred_ids = [9, 1]  # 9 is not in id2label -> falls back to "O"
    pred_scores = [0.99, 0.99]

    mentions = _bio_to_mentions(
        text=text,
        offsets=offsets,
        word_ids=word_ids,
        pred_ids=pred_ids,
        pred_scores=pred_scores,
        id2label=_id2label(),
        min_confidence=0.5,
    )
    # Only "Bar" survives as B-PER.
    assert [m.surface_form for m in mentions] == ["Bar"]


def test_bio_to_mentions_empty_input() -> None:
    assert (
        _bio_to_mentions(
            text="",
            offsets=[],
            word_ids=[],
            pred_ids=[],
            pred_scores=[],
            id2label=_id2label(),
            min_confidence=0.5,
        )
        == []
    )


def test_bio_to_mentions_b_after_b_starts_new_span() -> None:
    """Two adjacent B-PER tokens (different entities) become two mentions."""
    text = "Alice Bob"
    offsets = [(0, 5), (6, 9)]
    word_ids = [0, 1]
    pred_ids = [1, 1]  # both B-PER
    pred_scores = [0.99, 0.99]

    mentions = _bio_to_mentions(
        text=text,
        offsets=offsets,
        word_ids=word_ids,
        pred_ids=pred_ids,
        pred_scores=pred_scores,
        id2label=_id2label(),
        min_confidence=0.5,
    )
    assert [m.surface_form for m in mentions] == ["Alice", "Bob"]


# ---------------------------------------------------------------------------
# OnnxExtractor.extract — tested with fake onnxruntime session and tokenizer
# so we don't need the real ONNX model on disk in unit CI.
# ---------------------------------------------------------------------------


class _FakeEncoding(dict):  # type: ignore[type-arg]
    def __init__(self, ids, mask, offsets, word_ids):  # type: ignore[no-untyped-def]
        super().__init__()
        self["input_ids"] = ids
        self["attention_mask"] = mask
        self["offset_mapping"] = offsets
        self._word_ids = word_ids

    def word_ids(self, batch_index: int = 0) -> list[int | None]:
        return self._word_ids


class _FakeTokenizer:
    def __call__(self, text, **kwargs):  # type: ignore[no-untyped-def]
        import numpy as np

        # Return one batch with 4 tokens: [CLS] Alice Bob [SEP]
        ids = np.array([[101, 1, 2, 102]], dtype=np.int64)
        mask = np.array([[1, 1, 1, 1]], dtype=np.int64)
        offsets = np.array([[[0, 0], [0, 5], [6, 9], [0, 0]]], dtype=np.int64)
        return _FakeEncoding(ids, mask, offsets, [None, 0, 1, None])


class _FakeSession:
    def run(self, names, feeds):  # type: ignore[no-untyped-def]
        import numpy as np

        # logits for 4 tokens, 9 labels: B-PER on tokens 1 + 2.
        logits = np.full((1, 4, 9), -10.0, dtype=np.float32)
        logits[0, 0, 0] = 10.0  # CLS -> O
        logits[0, 1, 1] = 10.0  # Alice -> B-PER
        logits[0, 2, 1] = 10.0  # Bob -> B-PER
        logits[0, 3, 0] = 10.0  # SEP -> O
        return [logits]


def test_onnx_extractor_extract_with_injected_session() -> None:
    from ner_pipeline.onnx_extractor import OnnxExtractor

    ext = OnnxExtractor("nonexistent.onnx", tokenizer_name="dslim/distilbert-NER")
    # Inject the fakes so we don't hit the disk / network.
    ext._sess = _FakeSession()
    ext._tokenizer = _FakeTokenizer()
    ext._id2label = _id2label()

    out = ext.extract("Alice Bob")
    assert [(m.surface_form, m.entity_type) for m in out] == [
        ("Alice", "PER"),
        ("Bob", "PER"),
    ]


def test_onnx_extractor_empty_text() -> None:
    from ner_pipeline.onnx_extractor import OnnxExtractor

    ext = OnnxExtractor("nonexistent.onnx")
    assert ext.extract("") == []
    assert ext.extract("   ") == []
