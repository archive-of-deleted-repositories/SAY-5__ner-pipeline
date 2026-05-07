"""Smoke test: fit Platt scaling on a small CoNLL slice and assert Brier
score improves on the held-out tail.

Skipped unless ``RUN_EVAL_SMOKE=1``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.eval_smoke


def test_calibrate_smoke_improves_brier(tmp_path: Path) -> None:
    from scripts.calibrate import main

    params_path = tmp_path / "platt.json"
    brier_path = tmp_path / "brier.json"
    rc = main(
        [
            "--limit",
            "200",
            "--out-params",
            str(params_path),
            "--out-brier",
            str(brier_path),
        ]
    )
    assert rc == 0
    assert params_path.exists()
    with brier_path.open() as f:
        report = json.load(f)
    print(
        f"\nbrier overall: raw={report['before']['overall']['raw']:.4f} "
        f"calibrated={report['after']['overall']['calibrated']:.4f}"
    )
    raw = float(report["before"]["overall"]["raw"])
    cal = float(report["after"]["overall"]["calibrated"])
    # Calibration should reduce Brier (or at worst tie within numerical noise).
    assert cal <= raw + 1e-6
