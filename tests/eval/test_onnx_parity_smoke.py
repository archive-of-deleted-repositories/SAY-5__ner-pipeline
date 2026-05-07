"""Smoke parity test: export the live model to ONNX and assert PT/ONNX agree.

Skipped unless ``RUN_EVAL_SMOKE=1``. Runs in the ``onnx-parity-smoke`` CI job.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.eval_smoke


# Per-token max abs diff between PT logits and ONNX logits must stay below
# this. dslim/distilbert-NER on CPU is comfortably below 1e-4 in practice.
PARITY_TOL = 1.0e-4


def test_onnx_parity_20_sentences(tmp_path: Path) -> None:
    """Export the model + run parity_check.run_parity over 20 CoNLL sentences."""
    from scripts.export_onnx import export
    from scripts.parity_check import run_parity

    onnx_path = tmp_path / "distilbert_ner.onnx"
    export("dslim/distilbert-NER", onnx_path)

    report = run_parity(
        model_name="dslim/distilbert-NER",
        onnx_path=onnx_path,
        n=20,
        tol=PARITY_TOL,
    )
    print(f"\nparity max_abs_diff={report.max_abs_diff:.3e} agreement={report.label_agreement:.4f}")
    assert report.label_agreement >= 0.999
    assert report.max_abs_diff < PARITY_TOL
    assert report.pass_ is True
    # Spot-check the sidecar exists and is valid JSON.
    sidecar = onnx_path.with_suffix(".meta.json")
    assert sidecar.exists()
    with sidecar.open() as f:
        meta = json.load(f)
    assert meta["model_name"] == "dslim/distilbert-NER"
    assert meta["num_labels"] == 9
