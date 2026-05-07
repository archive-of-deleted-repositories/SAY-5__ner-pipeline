# Eval baselines

This directory holds committed metrics from real eval runs. The README in the
project root cites the numbers in [`conll_test_distilbert.json`](./conll_test_distilbert.json).

## conll_test_distilbert.json

| Field | Value |
|---|---|
| Model | `dslim/distilbert-NER` |
| Dataset | `eriktks/conll2003`, `test` split |
| Sentences evaluated | 3 453 |
| Hardware | Apple M-series (CPU only) |
| Run date | 2026-05-07 |

### Methodology

The extractor is the same one used in production: text in, char-span mentions
out (HuggingFace `pipeline("ner", aggregation_strategy="first")`).

For eval the per-sentence CoNLL tokens are joined with single spaces to form
input text; predicted char-spans are then aligned back to whole-token spans;
seqeval scores at the entity level (BIO).

This means the reported number is the F1 you actually get from the production
pathway end-to-end — **not** the model's headline number against pre-tokenised
input. The two differ because joining tokens with single spaces drops the
original punctuation context. See `ARCHITECTURE.md` for the trade-off.

### Reproduce

```bash
make eval                                # writes a fresh result JSON
python scripts/eval_conll.py --limit 100 # 100-sentence subset
```

### Smoke gate

`tests/eval/test_eval_smoke.py` runs the same evaluator on the first 100
sentences and asserts F1 ≥ 0.85. The `eval-smoke` CI job runs this test.
