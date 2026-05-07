"""Export the live HF NER model to ONNX with dynamic batch + sequence axes.

Usage:
    python scripts/export_onnx.py \
        --model dslim/distilbert-NER \
        --out artifacts/distilbert_ner.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def export(model_name: str, out_path: Path, opset: int = 14) -> Path:
    """Export ``model_name`` to ONNX with dynamic batch + sequence axes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    model.eval()

    # Build a representative dummy batch with attention_mask so the export
    # captures both inputs. Sequence length is dynamic at runtime; the dummy
    # length only seeds the trace.
    dummy = tokenizer(
        ["The Eiffel Tower is in Paris."],
        return_tensors="pt",
        padding="max_length",
        max_length=32,
        truncation=True,
    )
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    }

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    # Validate the on-disk graph is well-formed.
    model_proto = onnx.load(str(out_path))
    onnx.checker.check_model(model_proto)

    # Persist a small sidecar with the label map so consumers can
    # decode logits without re-loading the HF config.
    sidecar = {
        "model_name": model_name,
        "opset": opset,
        "id2label": model.config.id2label,
        "label2id": model.config.label2id,
        "num_labels": model.config.num_labels,
        "input_names": input_names,
        "output_names": output_names,
    }
    sidecar_path = out_path.with_suffix(".meta.json")
    with sidecar_path.open("w") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True, default=str)

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dslim/distilbert-NER")
    parser.add_argument("--out", type=Path, default=Path("artifacts/distilbert_ner.onnx"))
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()
    path = export(args.model, args.out, opset=args.opset)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"exported {args.model} -> {path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
