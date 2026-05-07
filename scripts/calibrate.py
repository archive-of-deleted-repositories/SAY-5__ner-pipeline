"""Fit per-entity-type Platt scaling on the CoNLL-2003 train split.

Workflow:

1. Load CoNLL-2003 train split, run the live extractor over each sentence.
2. For each predicted mention, label it ``1`` (true positive) if its
   token-aligned span matches a gold span of the same type, else ``0``.
3. Fit a separate Platt sigmoid per entity type; write
   ``artifacts/calibration_platt.json``.
4. Evaluate Brier-score before and after on a held-out tail of train; emit
   ``artifacts/calibration_brier.json`` for the README.

This is hermetic-by-default: pass ``--limit 200`` to fit on a small subset
in CI smoke jobs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ner_pipeline.calibration import (
    CalibrationParams,
    apply_platt,
    brier_score,
    fit_calibration,
)
from ner_pipeline.eval import (
    align_char_mentions_to_tokens,
    build_token_offsets,
    labels_from_ids,
)
from ner_pipeline.extractor import ENTITY_TYPES, Extractor


def _gold_spans(tokens: list[str], gold_ids: list[int]) -> set[tuple[int, int, str]]:
    """Convert a CoNLL BIO tag sequence into a set of (start, end, type) triples."""
    labels = labels_from_ids(gold_ids)
    out: set[tuple[int, int, str]] = set()
    i = 0
    while i < len(labels):
        tag = labels[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{etype}":
                j += 1
            out.add((i, j, etype))
            i = j
        else:
            i += 1
    return out


def collect_samples(
    extractor: Extractor,
    sentences: list[list[str]],
    gold_label_ids: list[list[int]],
) -> dict[str, list[tuple[float, int]]]:
    """For each predicted mention, emit (raw_confidence, label∈{0,1}) per type.

    A prediction is positive (label=1) iff its token-aligned span equals a
    gold span of the same type; else label=0. Gold spans never matched by a
    prediction are missed-recall events and don't contribute to Platt fits
    (Platt is calibrating *predictions*, not flagging missed entities).
    """
    samples: dict[str, list[tuple[float, int]]] = {et: [] for et in ENTITY_TYPES}
    for tokens, gold_ids in zip(sentences, gold_label_ids, strict=True):
        text, offsets = build_token_offsets(tokens)
        char_mentions = [
            (m.char_start, m.char_end, m.entity_type, m.confidence) for m in extractor.extract(text)
        ]
        token_mentions = align_char_mentions_to_tokens(
            tokens, text, offsets, [(s, e, t) for s, e, t, _ in char_mentions]
        )
        gold = _gold_spans(tokens, gold_ids)
        for (start, end, etype), (_, _, _, conf) in zip(
            token_mentions, char_mentions, strict=False
        ):
            label = 1 if (start, end, etype) in gold else 0
            if etype in samples:
                samples[etype].append((float(conf), label))
    return samples


def _split(samples: dict[str, list[tuple[float, int]]], frac: float) -> tuple[
    dict[str, list[tuple[float, int]]],
    dict[str, list[tuple[float, int]]],
]:
    """Deterministic head/tail split per type at index ``floor(n*frac)``."""
    train: dict[str, list[tuple[float, int]]] = defaultdict(list)
    test: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for etype, pairs in samples.items():
        cut = int(len(pairs) * frac)
        train[etype] = pairs[:cut]
        test[etype] = pairs[cut:]
    return dict(train), dict(test)


def evaluate_brier(
    samples: dict[str, list[tuple[float, int]]],
    params: CalibrationParams | None,
) -> dict[str, dict[str, float]]:
    """Compute per-type Brier score before vs. after calibration."""
    out: dict[str, dict[str, float]] = {}
    all_raw_p: list[float] = []
    all_cal_p: list[float] = []
    all_y: list[int] = []
    for etype, pairs in samples.items():
        if not pairs:
            out[etype] = {"raw": 0.0, "calibrated": 0.0, "n": 0.0}
            continue
        raw_p = [p for p, _ in pairs]
        y = [yi for _, yi in pairs]
        if params is None:
            cal_p = list(raw_p)
        else:
            pp = params.per_type.get(etype, params.fallback)
            cal_p = [apply_platt(p, pp) for p in raw_p]
        out[etype] = {
            "raw": brier_score(raw_p, y),
            "calibrated": brier_score(cal_p, y),
            "n": float(len(pairs)),
        }
        all_raw_p.extend(raw_p)
        all_cal_p.extend(cal_p)
        all_y.extend(y)
    out["overall"] = {
        "raw": brier_score(all_raw_p, all_y),
        "calibrated": brier_score(all_cal_p, all_y),
        "n": float(len(all_y)),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument(
        "--out-params",
        type=Path,
        default=Path("artifacts/calibration_platt.json"),
    )
    parser.add_argument(
        "--out-brier",
        type=Path,
        default=Path("artifacts/calibration_brier.json"),
    )
    parser.add_argument("--model", default="dslim/distilbert-NER")
    args = parser.parse_args(argv)

    from datasets import load_dataset  # type: ignore[import-not-found]

    from ner_pipeline.extractor import TransformersExtractor

    ds = load_dataset("conll2003", split="train", trust_remote_code=True)
    n = min(args.limit, len(ds))
    sentences = [list(r["tokens"]) for r in ds.select(range(n))]
    gold = [list(r["ner_tags"]) for r in ds.select(range(n))]

    extractor = TransformersExtractor(model_name=args.model)
    samples = collect_samples(extractor, sentences, gold)

    train, test = _split(samples, args.train_frac)
    params = fit_calibration(train)
    params.save(args.out_params)

    brier_before = evaluate_brier(test, params=None)
    brier_after = evaluate_brier(test, params=params)
    payload: dict[str, Any] = {
        "n_train_total": sum(len(v) for v in train.values()),
        "n_test_total": sum(len(v) for v in test.values()),
        "model": args.model,
        "before": brier_before,
        "after": brier_after,
        "improvement_overall": (
            brier_before["overall"]["raw"] - brier_after["overall"]["calibrated"]
        ),
    }
    args.out_brier.parent.mkdir(parents=True, exist_ok=True)
    with args.out_brier.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
