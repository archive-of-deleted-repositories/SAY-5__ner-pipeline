"""Entity extractor protocol and implementations.

The Protocol is intentionally narrow: a single ``extract(text) -> list[Mention]``
call. Concrete implementations:

* :class:`TransformersExtractor` wraps a HuggingFace ``pipeline("ner", ...)`` and
  maps the model's BIO labels onto the closed enum used by the data model.
* :class:`MockExtractor` returns scripted mentions for deterministic unit tests
  without loading any model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

EntityType = Literal["PER", "ORG", "LOC", "MISC"]
ENTITY_TYPES: tuple[EntityType, ...] = ("PER", "ORG", "LOC", "MISC")


@dataclass(frozen=True)
class Mention:
    """A single entity mention spotted inside a source text.

    Char offsets are half-open: ``text[char_start:char_end] == surface_form``.
    """

    surface_form: str
    char_start: int
    char_end: int
    entity_type: EntityType
    confidence: float

    def __post_init__(self) -> None:
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError(f"invalid offsets: start={self.char_start}, end={self.char_end}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of [0,1]: {self.confidence}")
        if self.entity_type not in ENTITY_TYPES:
            raise ValueError(f"unknown entity_type: {self.entity_type!r}")
        if not self.surface_form:
            raise ValueError("surface_form must be non-empty")


@runtime_checkable
class Extractor(Protocol):
    """Anything that turns text into a list of ``Mention``."""

    def extract(self, text: str) -> list[Mention]: ...


# ---------------------------------------------------------------------------
# HF label mapping
# ---------------------------------------------------------------------------

# dslim/distilbert-NER and dslim/bert-base-NER both emit BIO labels over
# {PER, ORG, LOC, MISC}. ``aggregation_strategy="simple"`` strips the BIO
# prefix and groups subword tokens, leaving an ``entity_group`` like ``"PER"``.
_HF_GROUP_TO_ENUM: dict[str, EntityType] = {
    "PER": "PER",
    "PERSON": "PER",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "LOC": "LOC",
    "LOCATION": "LOC",
    "MISC": "MISC",
}


def map_hf_label(group: str) -> EntityType | None:
    """Map a HF ``entity_group`` onto our closed enum, or ``None`` to drop it.

    Unknown groups are dropped on purpose — silent enum widening would corrupt
    the downstream schema.
    """
    return _HF_GROUP_TO_ENUM.get(group.upper())


# ---------------------------------------------------------------------------
# Mock extractor (unit tests)
# ---------------------------------------------------------------------------


class MockExtractor:
    """Returns a fixed list of mentions regardless of input.

    Tests can also pre-program responses keyed by exact text match.
    """

    def __init__(
        self,
        default: list[Mention] | None = None,
        scripted: dict[str, list[Mention]] | None = None,
    ) -> None:
        self._default = default or []
        self._scripted = scripted or {}
        self.calls: list[str] = []

    def extract(self, text: str) -> list[Mention]:
        self.calls.append(text)
        if text in self._scripted:
            return list(self._scripted[text])
        return list(self._default)


# ---------------------------------------------------------------------------
# Transformers extractor (real model)
# ---------------------------------------------------------------------------


class TransformersExtractor:
    """HuggingFace-backed NER extractor.

    The model is loaded lazily on first call so that import-time is cheap and
    unit tests can construct the object without network access (as long as
    they never call ``extract``).
    """

    def __init__(
        self,
        model_name: str = "dslim/distilbert-NER",
        revision: str | None = None,
        device: str = "cpu",
        min_confidence: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.min_confidence = min_confidence
        self._pipe: Any | None = None

    def _ensure_pipeline(self) -> Any:
        if self._pipe is None:
            # Imports are deferred so unit tests don't pay the torch / transformers
            # import cost (and don't need them installed at module-import time
            # if they only use MockExtractor).
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )

            tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call,unused-ignore]
                self.model_name, revision=self.revision
            )
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name, revision=self.revision
            )
            # ``first`` aggregates BIO tokens into whole-word spans by taking
            # the label of each word's first subword. ``simple`` was tempting
            # but produced fragmented sub-word spans on capitalised words at
            # sentence-internal positions — see ARCHITECTURE.md.
            # transformers' typed overloads don't expose the legacy ``"ner"``
            # task alias, but it is still supported at runtime.
            self._pipe = pipeline(  # type: ignore[call-overload,unused-ignore]
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="first",
                device=-1 if self.device == "cpu" else 0,
            )
        return self._pipe

    def extract(self, text: str) -> list[Mention]:
        if not text or not text.strip():
            return []
        pipe = self._ensure_pipeline()
        raw_mentions: list[dict[str, Any]] = pipe(text)

        out: list[Mention] = []
        for raw in raw_mentions:
            score = float(raw.get("score", 0.0))
            if score < self.min_confidence:
                continue
            etype = map_hf_label(str(raw.get("entity_group", "")))
            if etype is None:
                continue
            start = int(raw["start"])
            end = int(raw["end"])
            surface = text[start:end]
            if not surface:
                continue
            out.append(
                Mention(
                    surface_form=surface,
                    char_start=start,
                    char_end=end,
                    entity_type=etype,
                    confidence=score,
                )
            )
        return out
