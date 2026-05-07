# ner-pipeline

NLP entity-extraction pipeline for structured ingestion: a pretrained
transformer NER model reads unstructured text, classifies named entities
(`PER` / `ORG` / `LOC` / `MISC`), and writes deduplicated, char-offset-
preserving records into Postgres. Built around an `Extractor` Protocol so the
real model can be swapped for a `MockExtractor` in tests; ingestion is
idempotent on a SHA-256 of the source text.

## Eval results

CoNLL-2003 test split, `dslim/distilbert-NER`, run on Apple M-series (CPU).
See [`eval/baselines/conll_test_distilbert.json`](eval/baselines/conll_test_distilbert.json)
for the raw numbers and `make eval` to reproduce.

| Metric | Score |
|---|---|
| **F1**       | **0.8794** |
| Precision    | 0.8719 |
| Recall       | 0.8870 |
| Sentences    | 3 453 |
| Entity spans | 5 648 |

| Type | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| PER  | 0.9176 | 0.9227 | **0.9201** | 1 617 |
| ORG  | 0.8489 | 0.8627 | **0.8558** | 1 661 |
| LOC  | 0.9034 | 0.9191 | **0.9111** | 1 668 |
| MISC | 0.7510 | 0.7863 | **0.7683** | 702 |

These are the F1 numbers you get from the **production pathway end-to-end**
(text in → extractor → BIO alignment → seqeval). They differ from the
model card's 91.84% because that number is computed against pre-tokenised
input via `is_split_into_words=True`. The methodology trade-off is in
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## What this studies

* **HF-label → closed-enum mapping.** The model emits BIO labels over an open
  set; the schema needs a closed `{PER, ORG, LOC, MISC}` enum. Unknown labels
  are dropped on purpose — silent enum widening would corrupt the downstream
  table.
* **Char-offset preservation under WordPiece.** The extractor returns char
  spans into the original text, never WordPiece indices, so consumers can do
  highlighting / redaction without re-tokenising. The eval realigns those char
  spans onto the dataset's whitespace token stream.
* **Idempotent ingest under content-hash dedup.** A second call with the same
  text returns the existing `document_id` and writes nothing — the extractor
  is not even invoked. Entity rows are deduplicated by
  `(canonical_form, entity_type)` across documents; mention rows keep their
  per-occurrence char span and confidence.
* **Aggregation strategy matters.** Using `aggregation_strategy="simple"` on
  this model fragments capitalised words at sentence-internal positions
  (e.g. "Nadim" → `["Na", "di", "m"]`); `"first"` recovers the whole-word
  spans. This single-line difference moves overall F1 from 0.77 to 0.88 on
  the same data.

## Modules

| Module | Purpose |
|---|---|
| `ner_pipeline.extractor` | `Extractor` Protocol; `TransformersExtractor` (real model); `MockExtractor` (tests). |
| `ner_pipeline.pipeline`  | `ingest()` — content-hash dedup, entity dedup, mention persistence. |
| `ner_pipeline.models`    | SQLAlchemy ORM: `Document`, `Entity`, `EntityMention`. |
| `ner_pipeline.api`       | FastAPI: `POST /v1/ingest`, `/healthz`, error envelope. |
| `ner_pipeline.eval`      | CoNLL-2003 evaluator; BIO realignment; seqeval scoring. |
| `ner_pipeline.settings`  | Pydantic Settings (env-driven config). |
| `ner_pipeline.db`        | SQLAlchemy engine / session lifecycle. |
| `scripts/eval_conll.py`  | CLI: download CoNLL-2003 → run extractor → write JSON. |
| `scripts/seed.py`        | CLI: insert N synthetic Faker-driven documents. |

## Quickstart

```bash
make install
make up           # docker compose up postgres
make migrate      # alembic upgrade head
make seed         # 100 synthetic docs (uses the real model — slow first time)
make eval         # full CoNLL test split → eval/baselines/*.json
make serve        # uvicorn on :8000
```

`POST` example:

```bash
curl -X POST http://localhost:8000/v1/ingest \
  -H 'content-type: application/json' \
  -d '{"source":"demo","text":"Alice met Bob at Acme HQ in Berlin."}'
```

## Architecture

```
                                                      +-------------+
   text  ──►  TransformersExtractor  ──► [Mention]──► |  pipeline   |
                (lazy-loaded HF NER)                  |  ingest()   |
                                                      +------+------+
                                                             │
                                  content_sha256 dedup ──────┤
                                  entity dedup     ──────────┤
                                                             ▼
                                              +-------------------------+
                                              |       PostgreSQL        |
                                              |  documents              |
                                              |  entities               |
                                              |  entity_mentions        |
                                              +-------------------------+
```

## Tests

| Suite | Count | Default | Gate |
|---|---:|---|---|
| Unit (`tests/unit/`)            | 42 | always | `pytest tests/unit` |
| Integration (`tests/integration/`) |  5 | gated  | `RUN_INTEGRATION=1` (CI sets this; needs Postgres) |
| Eval-smoke (`tests/eval/`)       |  1 | gated  | `RUN_EVAL_SMOKE=1` (CI runs in its own job) |

Coverage gate: 80% on `src/ner_pipeline/` (current: 86%).

## Layout

```
src/ner_pipeline/    # package
alembic/             # migrations (0001_initial_schema)
scripts/             # eval_conll.py, seed.py
tests/{unit,integration,eval}/
eval/baselines/      # committed metrics from real runs
.github/workflows/   # ci.yml: lint, typecheck, test (3.11+3.12), eval-smoke, build
```

## What this is *not*

* No fine-tuning loop. The model is pretrained (`dslim/distilbert-NER`) and
  used as-is.
* No entity linking. Canonicalisation is whitespace-collapse + lowercase;
  there is no resolution against any knowledge base.
* No coreference resolution. "Alice" and "she" become separate mentions if
  the model emits them; only `she` won't be tagged because distilbert-NER
  doesn't predict pronouns as PER.
* No nested entities. The schema and BIO format both assume flat spans.
* No streaming inference. The endpoint is single-shot per request; long
  documents go in one piece (no chunking).
* No batched ingest endpoint. One document per call.

## License

MIT. See [`LICENSE`](LICENSE).
