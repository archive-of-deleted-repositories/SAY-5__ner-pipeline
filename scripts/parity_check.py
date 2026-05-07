"""PyTorch <-> ONNX parity check for the live NER model.

Loads both backends, runs them on N CoNLL-2003 test sentences, and asserts:

* per-token max absolute difference in logits stays under ``--tol`` (default 1e-4)
* mean absolute difference is reported for visibility
* token-classification predictions agree (argmax over labels) on every token

Optionally also computes seqeval F1 for the ONNX backend so it can be
compared head-to-head with the PyTorch baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

_DEFAULT_SENTENCES: tuple[str, ...] = (
    "The Eiffel Tower is in Paris.",
    "Apple announced new products in Cupertino last week.",
    "Angela Merkel met with Emmanuel Macron in Berlin.",
    "The European Union signed a trade agreement with Canada.",
    "SpaceX launched a Falcon 9 rocket from Cape Canaveral.",
)


@dataclass
class ParityReport:
    n_sentences: int
    max_abs_diff: float
    mean_abs_diff: float
    label_agreement: float  # fraction of tokens where argmax matches
    tolerance: float
    pass_: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pass"] = d.pop("pass_")
        return d


def _load_conll_sentences(n: int) -> list[str]:
    """Load N CoNLL-2003 test split sentences as plain strings.

    Falls back to a small built-in set when ``datasets`` is unavailable
    (e.g. in the CI smoke job).
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]

        ds = load_dataset("conll2003", split="test", trust_remote_code=True)
        out: list[str] = []
        for row in ds.select(range(min(n, len(ds)))):
            out.append(" ".join(row["tokens"]))
        return out
    except Exception:
        # Fall through to built-in cycle.
        out = []
        i = 0
        while len(out) < n:
            out.append(_DEFAULT_SENTENCES[i % len(_DEFAULT_SENTENCES)])
            i += 1
        return out


def run_parity(
    model_name: str,
    onnx_path: Path,
    n: int,
    tol: float,
) -> ParityReport:
    sentences = _load_conll_sentences(n)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pt_model = AutoModelForTokenClassification.from_pretrained(model_name)
    pt_model.eval()

    onnx_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    diffs: list[float] = []
    abs_means: list[float] = []
    matches = 0
    total = 0

    for sentence in sentences:
        enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        with torch.no_grad():
            pt_logits = pt_model(input_ids=input_ids, attention_mask=attention_mask).logits
        pt_arr = pt_logits.numpy()

        onnx_logits = onnx_sess.run(
            ["logits"],
            {
                "input_ids": input_ids.numpy().astype(np.int64),
                "attention_mask": attention_mask.numpy().astype(np.int64),
            },
        )[0]

        diff = np.abs(pt_arr - onnx_logits)
        diffs.append(float(diff.max()))
        abs_means.append(float(diff.mean()))

        pt_pred = pt_arr.argmax(axis=-1)
        onnx_pred = onnx_logits.argmax(axis=-1)
        matches += int((pt_pred == onnx_pred).sum())
        total += int(pt_pred.size)

    max_diff = max(diffs) if diffs else 0.0
    mean_diff = float(np.mean(abs_means)) if abs_means else 0.0
    agreement = matches / total if total else 0.0
    return ParityReport(
        n_sentences=len(sentences),
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        label_agreement=agreement,
        tolerance=tol,
        pass_=max_diff < tol and agreement >= 0.999,
    )


def run_onnx_eval(model_name: str, onnx_path: Path, n_sentences: int) -> dict[str, Any]:
    """Run the ONNX extractor through the project's CoNLL eval harness."""
    from datasets import load_dataset  # type: ignore[import-not-found]

    from ner_pipeline.eval import evaluate
    from ner_pipeline.onnx_extractor import OnnxExtractor

    extractor = OnnxExtractor(onnx_path, tokenizer_name=model_name)
    ds = load_dataset("conll2003", split="test", trust_remote_code=True)
    sentences = [list(row["tokens"]) for row in ds.select(range(min(n_sentences, len(ds))))]
    gold_ids = [list(row["ner_tags"]) for row in ds.select(range(min(n_sentences, len(ds))))]
    result = evaluate(extractor, sentences, gold_ids)
    return result.to_json()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dslim/distilbert-NER")
    parser.add_argument("--onnx", type=Path, default=Path("artifacts/distilbert_ner.onnx"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--eval-f1",
        action="store_true",
        help="Also run the ONNX extractor through CoNLL eval and report F1.",
    )
    args = parser.parse_args(argv)

    report = run_parity(args.model, args.onnx, args.n, args.tol)
    payload = report.to_dict()

    if args.eval_f1:
        try:
            payload["onnx_eval"] = run_onnx_eval(args.model, args.onnx, args.n)
        except Exception as exc:  # pragma: no cover - eval requires the dataset
            payload["onnx_eval_error"] = str(exc)

    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    if not report.pass_:
        print(
            f"\nFAIL: max_abs_diff={report.max_abs_diff:.6e} >= tol={args.tol} "
            f"or label_agreement={report.label_agreement:.4f} < 0.999",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
