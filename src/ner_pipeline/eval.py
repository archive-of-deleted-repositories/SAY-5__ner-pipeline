"""CoNLL-2003 evaluator.

We score at the entity level using ``seqeval``. Predictions from the extractor
(char-span based) are realigned onto the dataset's word-level token streams so
that BIO tagging is well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ner_pipeline.extractor import EntityType, Extractor

# CoNLL-2003 ner_tags integer -> BIO label, per the canonical dataset card.
CONLL_LABEL_NAMES: tuple[str, ...] = (
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
)


@dataclass
class EvalResult:
    """Aggregate eval metrics returned by :func:`evaluate`."""

    precision: float
    recall: float
    f1: float
    support: int
    per_type: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "support": self.support,
            "per_type": {
                k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in self.per_type.items()
            },
        }


def labels_from_ids(ids: list[int]) -> list[str]:
    """Translate CoNLL-2003 ``ner_tags`` integers into BIO label strings."""
    return [CONLL_LABEL_NAMES[i] for i in ids]


def predictions_to_bio(
    tokens: list[str],
    mentions: list[tuple[int, int, EntityType]],
) -> list[str]:
    """Convert a list of ``(token_start_idx, token_end_idx_exclusive, type)``
    predictions into a BIO label stream of the same length as ``tokens``.

    Overlapping predictions are resolved by first-write-wins (the earliest
    span is kept; later overlaps are dropped).
    """
    n = len(tokens)
    out = ["O"] * n
    taken = [False] * n
    for start, end, etype in mentions:
        if start < 0 or end > n or end <= start:
            continue
        if any(taken[start:end]):
            continue
        out[start] = f"B-{etype}"
        for i in range(start + 1, end):
            out[i] = f"I-{etype}"
        for i in range(start, end):
            taken[i] = True
    return out


def align_char_mentions_to_tokens(
    tokens: list[str],
    char_text: str,
    char_offsets: list[tuple[int, int]],
    char_mentions: list[tuple[int, int, EntityType]],
) -> list[tuple[int, int, EntityType]]:
    """Map char-span mentions onto whole-token spans.

    A mention covers token ``i`` if ``[char_start_i, char_end_i)`` overlaps the
    mention's char span. We expand to the smallest contiguous token range that
    fully contains the mention.
    """
    out: list[tuple[int, int, EntityType]] = []
    for cs, ce, etype in char_mentions:
        first = -1
        last = -1
        for i, (ts, te) in enumerate(char_offsets):
            if te <= cs or ts >= ce:
                continue
            if first == -1:
                first = i
            last = i
        if first == -1:
            continue
        out.append((first, last + 1, etype))
    return out


def build_token_offsets(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Reconstruct a single-string text and per-token char offsets by joining
    tokens with single spaces (CoNLL is whitespace-tokenised).
    """
    pieces: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            pieces.append(" ")
            cursor += 1
        offsets.append((cursor, cursor + len(tok)))
        pieces.append(tok)
        cursor += len(tok)
    return "".join(pieces), offsets


def evaluate(
    extractor: Extractor,
    sentences: list[list[str]],
    gold_label_ids: list[list[int]],
) -> EvalResult:
    """Run ``extractor`` over every sentence and score against gold BIO labels.

    Uses ``seqeval`` for entity-level P / R / F1 plus a per-type breakdown.
    """
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    if len(sentences) != len(gold_label_ids):
        raise ValueError("sentences / gold_label_ids length mismatch")

    gold_labels: list[list[str]] = [labels_from_ids(ids) for ids in gold_label_ids]
    pred_labels: list[list[str]] = []

    for tokens in sentences:
        text, offsets = build_token_offsets(tokens)
        char_mentions = [(m.char_start, m.char_end, m.entity_type) for m in extractor.extract(text)]
        token_mentions = align_char_mentions_to_tokens(tokens, text, offsets, char_mentions)
        pred_labels.append(predictions_to_bio(tokens, token_mentions))

    overall_p = float(precision_score(gold_labels, pred_labels, zero_division=0))
    overall_r = float(recall_score(gold_labels, pred_labels, zero_division=0))
    overall_f1 = float(f1_score(gold_labels, pred_labels, zero_division=0))

    report = classification_report(gold_labels, pred_labels, output_dict=True, zero_division=0)
    per_type: dict[str, dict[str, float]] = {}
    for label in ("PER", "ORG", "LOC", "MISC"):
        if label in report:
            row = report[label]
            per_type[label] = {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1-score"]),
                "support": float(row["support"]),
            }

    support = sum(1 for labels in gold_labels for tag in labels if tag.startswith("B-"))
    return EvalResult(
        precision=overall_p,
        recall=overall_r,
        f1=overall_f1,
        support=support,
        per_type=per_type,
    )
