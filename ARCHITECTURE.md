# Architecture

This document covers the design decisions behind `ner-pipeline` — the
trade-offs that aren't obvious from reading the code.

## Extractor Protocol vs. inheritance

`Extractor` is a `typing.Protocol`, not an abstract base class. Concrete
implementations don't inherit from anything — they just match the shape:

```python
class Extractor(Protocol):
    def extract(self, text: str) -> list[Mention]: ...
```

Why Protocol:

* **Decouples tests from the real model.** `MockExtractor` doesn't need to
  import torch or transformers; unit tests run in <2s without ever loading
  weights.
* **Decouples the data model from the inference engine.** Swapping
  `TransformersExtractor` for a vendor API client (or a regex-based stub for a
  bake-off) is a one-line change at the FastAPI dependency-injection seam.
* **Avoids the diamond inheritance trap** when downstream code wants to mix in
  caching or instrumentation.

The `@runtime_checkable` decorator lets `isinstance(x, Extractor)` work for
defensive checks at the API edge, but no code currently relies on it.

## HF-label → closed enum mapping

`dslim/distilbert-NER` emits BIO labels over an open string set. Our schema
uses a closed enum `{PER, ORG, LOC, MISC}`. The mapping table is
intentionally explicit:

| HF group                     | Closed enum |
|------------------------------|-------------|
| `PER`, `PERSON`              | `PER`       |
| `ORG`, `ORGANIZATION`        | `ORG`       |
| `LOC`, `LOCATION`            | `LOC`       |
| `MISC`                       | `MISC`      |
| anything else                | **dropped** |

Rationale: a future model swap (say, to one that also emits `DATE` or
`EVENT`) must not silently widen the schema. Unknown labels are dropped at
the mapping boundary so the database invariant — `entity_type IN
('PER','ORG','LOC','MISC')` — holds even when the model upstream changes.

## Char-offset preservation under WordPiece

Subword tokenisation is the obvious tax of using a transformer NER model.
The extractor returns char offsets into the **original text**, not token
indices, by relying on the HuggingFace pipeline's `start` / `end` keys.
That makes:

* **Highlighting / redaction** in downstream UIs trivial — slicing
  `text[char_start:char_end]` always yields the surface form.
* **Mention table layout** straightforward — store `(char_start, char_end,
  surface_form, confidence)` and never look at WordPiece pieces again.

The trickier consequence is at eval time. The CoNLL-2003 dataset hands you
**whitespace-tokenised** sentences with parallel BIO labels:

```
tokens = ["Nadim", "Ladki"]
ner_tags = [B-PER, I-PER]
```

Joining with single spaces gives `"Nadim Ladki"`. The model produces char
spans `(0, 5)` and `(6, 11)`. We have to map each char span back onto the
whole-token range `[0, 2)` so seqeval can score at the entity level. That
realignment lives in `ner_pipeline.eval.align_char_mentions_to_tokens`.

## Aggregation strategy: `first` over `simple`

HuggingFace's `pipeline("ner", aggregation_strategy=...)` has four
documented strategies. `"simple"` looked obviously right ("group adjacent
tokens with the same entity type") but produced wreckage on inputs like
`"Nadim Ladki"`:

```
[(0, 2, 'PER', 'Na'), (2, 4, 'PER', 'di'), (4, 11, 'PER', 'm Ladki')]
```

The subwords are predicted with the same label group but different scores,
and `simple` doesn't merge them across the WordPiece boundary in this case.
`"first"` ("use the first subword's label for the whole word, aggregate
under that label") fixes it:

```
[(0, 5, 'PER', 'Nadim'), (6, 11, 'PER', 'Ladki')]
```

This single-line change moves overall CoNLL-2003 test F1 from **0.7666 →
0.8794** on the same model and the same dataset. There is a unit-test floor
in `tests/eval/test_eval_smoke.py` to catch regressions if anyone reverts it.

## Idempotency: content-hash + entity dedup

Two layers of dedup, with different jobs.

**Document level — `content_sha256`.** A unique index on
`documents.content_sha256` (the SHA-256 of the UTF-8 raw text) means a
second `ingest()` with the same text:

* returns the existing `document_id`,
* does not re-invoke the extractor,
* does not write any new mention rows.

This protects against retries (network blips at the API layer) and
deliberate replays (re-running a backfill twice) without per-source dedup
logic.

**Entity level — `(canonical_form, entity_type)`.** The `entities` table has
a unique index on the pair. Two surface mentions of "Apple" in two different
documents share one `entities` row but get their own `entity_mentions` rows
with their own char spans and confidences.

`canonical_form` is whitespace-collapsed lowercase. That's deliberately
simple: a real entity-linker would do a lot more (alias tables, embedding
match against Wikidata, etc.) but those belong in a downstream service, not
in the ingest hot path.

## Eval methodology: seqeval, BIO, entity-level

We score with `seqeval`, which evaluates **entity-level** F1: a prediction
counts as a true positive only if the type and the **full span** match. A
`B-PER I-PER` predicted as `B-PER O` is one missed entity, not one TP and
one FN.

Per-type breakdown is reported separately because aggregate F1 hides where
errors are concentrated. In the committed baseline, MISC (0.768) is
materially worse than PER / ORG / LOC — unsurprising for this dataset
because MISC includes a long tail of nationalities, events, and works that
are noisier to label.

## Why CoNLL-2003 specifically

It's the canonical NER benchmark, comes pre-packaged in the `datasets`
library, and is small enough (3 453 test sentences) to evaluate on CI in a
few minutes. It also matches the `dslim/distilbert-NER` model's training
distribution, which keeps the eval scoring the model's intended use case
rather than its zero-shot generalisation.

## What's deliberately not here

* **A training loop.** The model is pretrained; there is no fine-tuning
  pipeline. Adding one means committing to a training-data layout, an
  experiment tracker, and reproducibility infra that's out of scope for a
  pipeline-focused codebase.
* **Coreference resolution.** "Alice waved. She smiled." would yield one PER
  mention, not two.
* **Nested entities.** The BIO tagging in the dataset and our schema both
  assume flat spans. Nested NER (e.g. "[University of [California]]")
  needs a different label format and a different model.
* **Streaming / chunked inference.** Long documents go through the pipeline
  in one shot. Chunking with overlap-and-merge would belong here if we hit
  context-window limits in production, but the test-set sentences fit
  comfortably.
* **Knowledge-base linking.** Canonical forms are not resolved against
  Wikidata, DBpedia, or any internal KB. That's a separate service.
