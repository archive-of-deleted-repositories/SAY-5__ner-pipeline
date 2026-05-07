"""Unit tests for the eval module's BIO conversion and metric computation."""

from __future__ import annotations

from ner_pipeline.eval import (
    align_char_mentions_to_tokens,
    build_token_offsets,
    evaluate,
    labels_from_ids,
    predictions_to_bio,
)
from ner_pipeline.extractor import Mention, MockExtractor


def test_labels_from_ids_canonical() -> None:
    # ner_tags ids: 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, ...
    assert labels_from_ids([0, 1, 2, 0, 5, 6]) == [
        "O",
        "B-PER",
        "I-PER",
        "O",
        "B-LOC",
        "I-LOC",
    ]


def test_predictions_to_bio_basic() -> None:
    tokens = ["Alice", "lives", "in", "Berlin"]
    out = predictions_to_bio(tokens, [(0, 1, "PER"), (3, 4, "LOC")])
    assert out == ["B-PER", "O", "O", "B-LOC"]


def test_predictions_to_bio_multi_token_span() -> None:
    tokens = ["New", "York", "is", "big"]
    out = predictions_to_bio(tokens, [(0, 2, "LOC")])
    assert out == ["B-LOC", "I-LOC", "O", "O"]


def test_predictions_to_bio_overlap_dropped() -> None:
    tokens = ["A", "B", "C"]
    out = predictions_to_bio(tokens, [(0, 2, "PER"), (1, 3, "ORG")])
    # Second mention overlaps the first — first wins.
    assert out == ["B-PER", "I-PER", "O"]


def test_predictions_to_bio_invalid_span_skipped() -> None:
    tokens = ["A", "B"]
    out = predictions_to_bio(tokens, [(-1, 1, "PER"), (0, 5, "PER"), (1, 1, "PER")])
    assert out == ["O", "O"]


def test_build_token_offsets_roundtrip() -> None:
    tokens = ["Hello", "World"]
    text, offsets = build_token_offsets(tokens)
    assert text == "Hello World"
    assert offsets == [(0, 5), (6, 11)]
    assert text[offsets[0][0] : offsets[0][1]] == "Hello"
    assert text[offsets[1][0] : offsets[1][1]] == "World"


def test_align_char_mentions_to_tokens() -> None:
    tokens = ["Alice", "visits", "New", "York"]
    text, offsets = build_token_offsets(tokens)
    # Char span over "New York"
    new_york_start = text.index("New")
    new_york_end = new_york_start + len("New York")
    aligned = align_char_mentions_to_tokens(
        tokens, text, offsets, [(new_york_start, new_york_end, "LOC")]
    )
    assert aligned == [(2, 4, "LOC")]


def test_align_char_mentions_partial_overlap_expands_to_full_tokens() -> None:
    tokens = ["Alice", "Smith"]
    text, offsets = build_token_offsets(tokens)
    # Span covers "lice Smit" — should expand to cover both whole tokens.
    aligned = align_char_mentions_to_tokens(tokens, text, offsets, [(1, 9, "PER")])
    assert aligned == [(0, 2, "PER")]


def test_evaluate_perfect_predictions() -> None:
    tokens = ["Alice", "in", "Berlin"]
    text, offsets = build_token_offsets(tokens)
    extractor = MockExtractor(
        scripted={
            text: [
                Mention("Alice", offsets[0][0], offsets[0][1], "PER", 0.99),
                Mention("Berlin", offsets[2][0], offsets[2][1], "LOC", 0.99),
            ]
        }
    )
    gold_ids = [[1, 0, 5]]  # B-PER O B-LOC
    result = evaluate(extractor, [tokens], gold_ids)
    assert result.f1 == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.support == 2


def test_evaluate_missed_entity_lowers_recall() -> None:
    tokens = ["Alice", "and", "Bob"]
    text, offsets = build_token_offsets(tokens)
    extractor = MockExtractor(
        scripted={text: [Mention("Alice", offsets[0][0], offsets[0][1], "PER", 0.99)]}
    )
    gold_ids = [[1, 0, 1]]  # B-PER O B-PER
    result = evaluate(extractor, [tokens], gold_ids)
    assert 0.5 < result.precision <= 1.0
    assert 0.0 < result.recall < 1.0
    assert result.f1 < 1.0
