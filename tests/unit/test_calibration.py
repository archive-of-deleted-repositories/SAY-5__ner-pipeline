"""Unit tests for Platt scaling calibration."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from ner_pipeline.calibration import (
    CalibrationParams,
    PlattParams,
    apply_platt,
    brier_score,
    calibrated_confidence,
    fit_calibration,
    fit_platt_one_type,
)
from ner_pipeline.extractor import Mention, MockExtractor
from ner_pipeline.models import EntityMention
from ner_pipeline.pipeline import ingest

# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def test_apply_platt_identity_when_a1_b0() -> None:
    pp = PlattParams(a=1.0, b=0.0, n_train=0)
    for p in (0.05, 0.5, 0.95):
        assert math.isclose(apply_platt(p, pp), p, abs_tol=1e-6)


def test_apply_platt_pushes_toward_zero_with_negative_bias() -> None:
    pp = PlattParams(a=1.0, b=-2.0, n_train=0)
    out = apply_platt(0.5, pp)
    assert out < 0.5


def test_apply_platt_pushes_toward_one_with_positive_bias() -> None:
    pp = PlattParams(a=1.0, b=2.0, n_train=0)
    out = apply_platt(0.5, pp)
    assert out > 0.5


def test_apply_platt_clips_extreme_inputs() -> None:
    pp = PlattParams(a=1.0, b=0.0, n_train=0)
    # Logit(0) is -inf; the implementation must clamp before sigmoid.
    out_lo = apply_platt(0.0, pp)
    out_hi = apply_platt(1.0, pp)
    assert 0.0 < out_lo < 0.5
    assert 0.5 < out_hi < 1.0


def test_brier_score_perfect_predictions() -> None:
    assert brier_score([0.0, 1.0, 0.0, 1.0], [0, 1, 0, 1]) == 0.0


def test_brier_score_worst_predictions() -> None:
    assert brier_score([1.0, 0.0], [0, 1]) == 1.0


def test_brier_score_constant_half() -> None:
    # All predictions = 0.5 with mixed labels -> 0.25.
    assert math.isclose(brier_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]), 0.25)


def test_brier_score_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        brier_score([0.5, 0.5], [1])


def test_brier_score_empty() -> None:
    assert brier_score([], []) == 0.0


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def test_fit_platt_one_type_empty() -> None:
    pp = fit_platt_one_type([], [])
    assert pp.a == 1.0 and pp.b == 0.0 and pp.n_train == 0


def test_fit_platt_one_type_length_mismatch() -> None:
    with pytest.raises(ValueError):
        fit_platt_one_type([0.5], [0, 1])


def test_fit_platt_overconfident_predictions_get_penalised() -> None:
    """When the model says 0.99 but is right only half the time, calibration
    should pull the probability **down**.
    """
    raw = [0.99] * 100
    labels = [1, 0] * 50  # 50% positive
    pp = fit_platt_one_type(raw, labels)
    out = apply_platt(0.99, pp)
    # Calibrated probability should be much closer to 0.5 than to 0.99.
    assert out < 0.85


def test_fit_calibration_per_type_keys_present() -> None:
    samples = {
        "PER": [(0.9, 1), (0.9, 1), (0.4, 0)],
        "ORG": [(0.8, 1), (0.3, 0)],
        "LOC": [(0.95, 1)],
        "MISC": [(0.5, 0)],
    }
    cal = fit_calibration(samples)
    assert set(cal.per_type.keys()) == {"PER", "ORG", "LOC", "MISC"}
    assert cal.fallback.n_train == 7


def test_fit_calibration_brier_improves_on_synthetic_overconfidence() -> None:
    """End-to-end: calibrated probabilities have lower Brier than raw."""
    # 200 predictions, all said with raw confidence 0.95, half wrong.
    raw_p = [0.95] * 200
    labels = [1, 0] * 100
    samples = {"PER": list(zip(raw_p, labels, strict=True))}
    cal = fit_calibration(samples)
    pp = cal.per_type["PER"]
    cal_p = [apply_platt(p, pp) for p in raw_p]
    assert brier_score(cal_p, labels) < brier_score(raw_p, labels)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_calibration_params_roundtrip(tmp_path: Path) -> None:
    samples = {
        "PER": [(0.9, 1), (0.4, 0)],
        "ORG": [(0.8, 1), (0.3, 0)],
        "LOC": [(0.95, 1), (0.6, 0)],
        "MISC": [(0.5, 0)],
    }
    cal = fit_calibration(samples)
    path = tmp_path / "platt.json"
    cal.save(path)
    loaded = CalibrationParams.load(path)
    for et in ("PER", "ORG", "LOC", "MISC"):
        assert math.isclose(loaded.per_type[et].a, cal.per_type[et].a, abs_tol=1e-9)
        assert math.isclose(loaded.per_type[et].b, cal.per_type[et].b, abs_tol=1e-9)
    assert math.isclose(loaded.fallback.a, cal.fallback.a, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def _identity_calibration() -> CalibrationParams:
    """Identity calibration: a=1, b=0 for every type."""
    pp = PlattParams(a=1.0, b=0.0, n_train=0)
    return CalibrationParams(per_type={t: pp for t in ("PER", "ORG", "LOC", "MISC")}, fallback=pp)


def test_calibrated_confidence_dispatches_per_type() -> None:
    pp_per = PlattParams(a=1.0, b=2.0, n_train=0)
    pp_fall = PlattParams(a=1.0, b=-2.0, n_train=0)
    cal = CalibrationParams(per_type={"PER": pp_per}, fallback=pp_fall)
    # PER uses the per-type scaler (positive bias -> > 0.5).
    assert calibrated_confidence(0.5, "PER", cal) > 0.5
    # ORG falls back to fallback (negative bias -> < 0.5).
    assert calibrated_confidence(0.5, "ORG", cal) < 0.5


def test_pipeline_ingest_applies_calibration_to_persisted_confidence(
    session: Session,
) -> None:
    extractor = MockExtractor(default=[Mention("Alice", 0, 5, "PER", 0.9)])
    # Strong negative bias for PER -> calibrated probability < raw.
    pp_per = PlattParams(a=1.0, b=-3.0, n_train=0)
    cal = CalibrationParams(per_type={"PER": pp_per}, fallback=PlattParams(a=1.0, b=0.0, n_train=0))
    res = ingest(
        source="t",
        text="Alice met Bob.",
        extractor=extractor,
        session=session,
        calibration=cal,
    )
    assert res.entity_count == 1
    persisted = session.query(EntityMention).one()
    # The persisted confidence has been pushed below the raw 0.9 by the bias.
    assert float(persisted.confidence) < 0.7


def test_pipeline_ingest_skips_calibration_when_none(session: Session) -> None:
    extractor = MockExtractor(default=[Mention("Alice", 0, 5, "PER", 0.9)])
    res = ingest(
        source="t",
        text="Alice met Bob.",
        extractor=extractor,
        session=session,
        calibration=None,
    )
    assert res.entity_count == 1
    persisted = session.query(EntityMention).one()
    # Raw 0.9, persisted as Numeric(5,4) so 0.9000 ± 1e-4.
    assert abs(float(persisted.confidence) - 0.9) < 1e-3


def test_pipeline_ingest_identity_calibration_preserves_raw(session: Session) -> None:
    extractor = MockExtractor(default=[Mention("Alice", 0, 5, "PER", 0.9)])
    res = ingest(
        source="t",
        text="hello",
        extractor=extractor,
        session=session,
        calibration=_identity_calibration(),
    )
    assert res.entity_count == 1
    persisted = session.query(EntityMention).one()
    assert abs(float(persisted.confidence) - 0.9) < 1e-3
