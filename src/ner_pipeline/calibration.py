"""Per-entity-type Platt scaling for confidence calibration.

HuggingFace NER returns the softmax score of the predicted label, which
is **not** a calibrated probability — well-known to be too sharp on
in-distribution and too flat on out-of-distribution inputs. Platt scaling
fits a 1-D logistic regression ``p_calibrated = sigmoid(a * logit_p + b)``
on a held-out split, separately per entity type, so downstream consumers
can treat ``EntityMention.confidence`` as an actual probability.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ner_pipeline.extractor import ENTITY_TYPES, EntityType


@dataclass(frozen=True)
class PlattParams:
    """Sigmoid parameters for a single entity type."""

    a: float
    b: float
    n_train: int


@dataclass(frozen=True)
class CalibrationParams:
    """Per-entity-type Platt parameters + identity fallback."""

    per_type: dict[str, PlattParams]
    fallback: PlattParams

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_type": {k: asdict(v) for k, v in self.per_type.items()},
            "fallback": asdict(self.fallback),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationParams:
        per_type = {k: PlattParams(**v) for k, v in payload["per_type"].items()}
        fallback = PlattParams(**payload["fallback"])
        return cls(per_type=per_type, fallback=fallback)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> CalibrationParams:
        with Path(path).open() as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def apply_platt(p: float, params: PlattParams) -> float:
    """Map a raw model score onto a calibrated probability."""
    return _sigmoid(params.a * _logit(p) + params.b)


def calibrated_confidence(
    raw_score: float,
    entity_type: EntityType | str,
    params: CalibrationParams,
) -> float:
    pp = params.per_type.get(str(entity_type), params.fallback)
    return apply_platt(raw_score, pp)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_platt_one_type(
    raw_scores: list[float],
    labels: list[int],
    *,
    lr: float = 0.05,
    epochs: int = 500,
    l2: float = 1e-4,
) -> PlattParams:
    """Fit ``p = sigmoid(a*logit(raw) + b)`` via gradient descent on log-loss.

    Pure-Python (no scipy / numpy required) so the fit is hermetic in CI.
    Convergence on the typical CoNLL-train-sized signal (a few thousand
    points) is fast at this LR; the L2 stops collapse when one class is rare.
    """
    if not raw_scores:
        return PlattParams(a=1.0, b=0.0, n_train=0)
    if len(raw_scores) != len(labels):
        raise ValueError("raw_scores and labels must have equal length")

    logits = [_logit(s) for s in raw_scores]
    n = len(logits)

    a, b = 1.0, 0.0
    for _ in range(epochs):
        ga = 0.0
        gb = 0.0
        for x, y in zip(logits, labels, strict=True):
            z = a * x + b
            p = _sigmoid(z)
            err = p - y
            ga += err * x
            gb += err
        ga = ga / n + l2 * a
        gb = gb / n + l2 * b
        a -= lr * ga
        b -= lr * gb

    return PlattParams(a=float(a), b=float(b), n_train=n)


def fit_calibration(
    samples_per_type: dict[str, list[tuple[float, int]]],
) -> CalibrationParams:
    """Fit a separate Platt scaler per entity type.

    ``samples_per_type[etype]`` is a list of ``(raw_score, label)`` pairs,
    where ``label`` is 1 if the prediction was a true positive and 0 if
    it was a false positive. The fallback scaler is fit on the union.
    """
    per_type: dict[str, PlattParams] = {}
    union_scores: list[float] = []
    union_labels: list[int] = []
    for etype in ENTITY_TYPES:
        pairs = samples_per_type.get(etype, [])
        scores = [p[0] for p in pairs]
        labels = [p[1] for p in pairs]
        per_type[etype] = fit_platt_one_type(scores, labels)
        union_scores.extend(scores)
        union_labels.extend(labels)
    fallback = fit_platt_one_type(union_scores, union_labels)
    return CalibrationParams(per_type=per_type, fallback=fallback)


# ---------------------------------------------------------------------------
# Brier score (mean squared error vs the binary outcome)
# ---------------------------------------------------------------------------


def brier_score(probs: list[float], labels: list[int]) -> float:
    if not probs:
        return 0.0
    if len(probs) != len(labels):
        raise ValueError("probs and labels must have equal length")
    return sum((p - y) ** 2 for p, y in zip(probs, labels, strict=True)) / len(probs)
