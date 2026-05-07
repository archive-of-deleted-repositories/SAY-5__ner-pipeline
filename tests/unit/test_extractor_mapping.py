"""Unit tests for HF-label-to-enum mapping and Mention validation."""

from __future__ import annotations

import pytest

from ner_pipeline.extractor import (
    ENTITY_TYPES,
    Mention,
    MockExtractor,
    map_hf_label,
)


class TestMapHfLabel:
    def test_known_labels(self) -> None:
        assert map_hf_label("PER") == "PER"
        assert map_hf_label("PERSON") == "PER"
        assert map_hf_label("ORG") == "ORG"
        assert map_hf_label("ORGANIZATION") == "ORG"
        assert map_hf_label("LOC") == "LOC"
        assert map_hf_label("LOCATION") == "LOC"
        assert map_hf_label("MISC") == "MISC"

    def test_case_insensitive(self) -> None:
        assert map_hf_label("per") == "PER"
        assert map_hf_label("Org") == "ORG"

    def test_unknown_label_returns_none(self) -> None:
        # An unknown HF label must drop the mention; it must NOT silently
        # widen the closed enum.
        assert map_hf_label("DATE") is None
        assert map_hf_label("EVENT") is None
        assert map_hf_label("") is None


class TestMentionValidation:
    def test_valid_mention(self) -> None:
        m = Mention(
            surface_form="Alice",
            char_start=0,
            char_end=5,
            entity_type="PER",
            confidence=0.99,
        )
        assert m.surface_form == "Alice"

    def test_negative_offsets_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid offsets"):
            Mention("x", -1, 5, "PER", 0.5)

    def test_inverted_offsets_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid offsets"):
            Mention("x", 5, 5, "PER", 0.5)

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Mention("x", 0, 1, "PER", 1.5)
        with pytest.raises(ValueError, match="confidence"):
            Mention("x", 0, 1, "PER", -0.1)

    def test_empty_surface_form_rejected(self) -> None:
        with pytest.raises(ValueError, match="surface_form"):
            Mention("", 0, 1, "PER", 0.9)

    def test_unknown_entity_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown entity_type"):
            Mention("x", 0, 1, "DATE", 0.9)  # type: ignore[arg-type]

    def test_all_enum_values_accepted(self) -> None:
        for et in ENTITY_TYPES:
            Mention("x", 0, 1, et, 0.5)


class TestMockExtractor:
    def test_default_response(self) -> None:
        m = Mention("Alice", 0, 5, "PER", 0.9)
        extractor = MockExtractor(default=[m])
        assert extractor.extract("anything") == [m]

    def test_scripted_response(self) -> None:
        m = Mention("Acme", 0, 4, "ORG", 0.8)
        extractor = MockExtractor(scripted={"hello": [m]})
        assert extractor.extract("hello") == [m]
        assert extractor.extract("other") == []

    def test_records_calls(self) -> None:
        extractor = MockExtractor()
        extractor.extract("first")
        extractor.extract("second")
        assert extractor.calls == ["first", "second"]
