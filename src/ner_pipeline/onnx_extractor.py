"""ONNX-Runtime-backed Extractor.

Loads an ONNX-exported NER model and reproduces the same char-span outputs
as :class:`ner_pipeline.extractor.TransformersExtractor`. Tokenisation still
uses the HuggingFace tokenizer (it has the offset-mapping logic), but
inference runs through ``onnxruntime`` for smaller memory footprint and
faster CPU latency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ner_pipeline.extractor import EntityType, Mention, map_hf_label


class OnnxExtractor:
    """Run inference via ``onnxruntime`` while keeping the same char-span API."""

    def __init__(
        self,
        onnx_path: str | Path,
        tokenizer_name: str = "dslim/distilbert-NER",
        meta_path: str | Path | None = None,
        min_confidence: float = 0.5,
        max_length: int = 256,
    ) -> None:
        self.onnx_path = str(onnx_path)
        self.tokenizer_name = tokenizer_name
        self.meta_path = Path(meta_path) if meta_path else Path(onnx_path).with_suffix(".meta.json")
        self.min_confidence = min_confidence
        self.max_length = max_length
        self._sess: Any | None = None
        self._tokenizer: Any | None = None
        self._id2label: dict[int, str] | None = None

    def _ensure_loaded(self) -> None:
        if self._sess is not None:
            return
        import numpy as np  # noqa: F401  (used downstream)
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self._sess = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        self._tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call,unused-ignore]
            self.tokenizer_name
        )

        if self.meta_path.exists():
            with self.meta_path.open() as f:
                meta = json.load(f)
            self._id2label = {int(k): str(v) for k, v in meta["id2label"].items()}
        else:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(self.tokenizer_name)
            self._id2label = {int(k): str(v) for k, v in cfg.id2label.items()}

    def extract(self, text: str) -> list[Mention]:
        if not text or not text.strip():
            return []
        self._ensure_loaded()
        assert self._sess is not None
        assert self._tokenizer is not None
        assert self._id2label is not None

        import numpy as np

        enc = self._tokenizer(
            text,
            return_tensors="np",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"].astype(np.int64)
        attention_mask = enc["attention_mask"].astype(np.int64)
        offsets: list[tuple[int, int]] = [tuple(o) for o in enc["offset_mapping"][0].tolist()]

        # word_ids() lets us collapse subwords back onto whole-word spans —
        # this mirrors HF's ``aggregation_strategy="first"``.
        word_ids = enc.word_ids(batch_index=0)

        logits = self._sess.run(
            ["logits"], {"input_ids": input_ids, "attention_mask": attention_mask}
        )[0]

        # Softmax over the label axis to get per-token probabilities.
        scores = _softmax(logits[0], axis=-1)
        pred_ids = scores.argmax(axis=-1).tolist()
        pred_scores = scores.max(axis=-1).tolist()

        # Group BIO labels into entity spans, mirroring HF "first" aggregation:
        # use only the first subword of each word, and extend its span out to
        # the end-offset of the last subword in the same word.
        return _bio_to_mentions(
            text=text,
            offsets=offsets,
            word_ids=word_ids,
            pred_ids=pred_ids,
            pred_scores=pred_scores,
            id2label=self._id2label,
            min_confidence=self.min_confidence,
        )


def _softmax(x: Any, axis: int = -1) -> Any:
    import numpy as np

    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def _bio_to_mentions(
    *,
    text: str,
    offsets: list[tuple[int, int]],
    word_ids: list[int | None],
    pred_ids: list[int],
    pred_scores: list[float],
    id2label: dict[int, str],
    min_confidence: float,
) -> list[Mention]:
    """Collapse subword tokens back onto whole-word spans, then group BIO.

    Mirrors HuggingFace's ``aggregation_strategy="first"``: each whole word
    inherits the label of its first subword, and consecutive same-type
    words form a single mention spanning their joint char range.
    """
    # Step 1: collapse subwords -> per-word (label_id, score, char_start, char_end).
    word_records: list[tuple[int, float, int, int]] = []
    last_word: int | None = None
    cur_word_end: int = -1
    cur_word_start: int = -1
    cur_label: int = 0
    cur_score: float = 0.0
    for tok_idx, wid in enumerate(word_ids):
        if wid is None:
            continue  # special tokens
        start, end = offsets[tok_idx]
        if wid != last_word:
            if last_word is not None:
                word_records.append((cur_label, cur_score, cur_word_start, cur_word_end))
            last_word = wid
            cur_word_start = start
            cur_word_end = end
            cur_label = pred_ids[tok_idx]
            cur_score = pred_scores[tok_idx]
        else:
            cur_word_end = end
    if last_word is not None:
        word_records.append((cur_label, cur_score, cur_word_start, cur_word_end))

    # Step 2: BIO -> mentions on the per-word stream.
    mentions: list[Mention] = []
    cur_type: EntityType | None = None
    cur_start = -1
    cur_end = -1
    cur_scores: list[float] = []

    def flush() -> None:
        nonlocal cur_type, cur_start, cur_end, cur_scores
        if cur_type is None or cur_start < 0:
            return
        surface = text[cur_start:cur_end]
        score = sum(cur_scores) / len(cur_scores) if cur_scores else 0.0
        if surface and score >= min_confidence:
            mentions.append(
                Mention(
                    surface_form=surface,
                    char_start=cur_start,
                    char_end=cur_end,
                    entity_type=cur_type,
                    confidence=min(max(score, 0.0), 1.0),
                )
            )
        cur_type = None
        cur_start = -1
        cur_end = -1
        cur_scores = []

    for label_id, score, w_start, w_end in word_records:
        label = id2label.get(label_id, "O")
        if label == "O":
            flush()
            continue
        if "-" in label:
            bio, group = label.split("-", 1)
        else:
            bio, group = "B", label
        etype = map_hf_label(group)
        if etype is None:
            flush()
            continue
        if bio == "B" or cur_type != etype:
            flush()
            cur_type = etype
            cur_start = w_start
            cur_end = w_end
            cur_scores = [score]
        else:
            cur_end = w_end
            cur_scores.append(score)
    flush()
    return mentions
