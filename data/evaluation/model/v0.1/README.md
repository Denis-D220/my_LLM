# Model Evaluation Set v0.1

Frozen held-out text for evaluating the trained Transformer.

## This is not the tokenizer evaluation set

`data/evaluation/tokenizer_eval.txt` and `data/evaluation/v0.2/` measure the
tokenizer: compression ratio, round-trip fidelity, byte fallback, case
preservation. Those properties are about encoding, and they say nothing about
whether a model learned anything.

This set measures the **model**: whether the trained network predicts held-out
technical English well. The two must stay separate. Reusing tokenizer
evaluation material here would also be unsafe, because that material was
selected to stress the encoder and is not representative of the corpus the
model actually trains on.

## Format

`pretraining_eval.jsonl` — one JSON object per line:

```json
{
  "id": "eval-engineering-0001",
  "category": "engineering",
  "text": "..."
}
```

| field | rule |
|---|---|
| `id` | unique across the file; loading rejects duplicates |
| `category` | free-form label used only for reporting |
| `text` | non-empty; items under 50 comparison tokens are skipped by the index |

Identifiers are stable. If an item's text changes, it gets a new identifier
rather than a silent edit, so a result recorded against `eval-physics-0001`
always refers to the same passage.

## Contents

20 items across 10 categories, two each:

```
general              software             engineering
physics              chemistry            biology
mathematics          technology           technical_manual
mixed
```

`mixed` contains prose interleaved with code and formulae, which is the
hardest case for a byte-level tokenizer and the one most likely to expose a
model that has only learned prose.

## Provenance

Every passage was written for this purpose. None is copied from the web, which
matters for two reasons. It keeps the set genuinely held out rather than
memorisable from any public source, and it means a decontamination audit that
reports zero hits against Common Crawl is the expected result rather than
evidence the detector works.

**Proving the detector works therefore requires planted controls** — real
documents taken from the corpus and fed through the decontamination index as
if they were evaluation items. See `scripts/audit_decontamination.py
--planted-controls`.

## Freeze policy

This set is frozen **before** the training corpus is finalised. That ordering
is not optional: deciding what to evaluate on after seeing what the corpus
contains invites selecting passages the model happens to handle well.

Changing any passage means a new version directory, `v0.2`, not an edit in
place. Results reported against v0.1 must remain reproducible.

## Known limitations of v0.1

This is a seed, sized to define the format and exercise the decontamination
pipeline, not a statistically adequate benchmark.

- **20 items is small.** Perplexity computed over roughly 30 KB of text carries
  meaningful variance. Expect to expand this considerably before comparing
  models that differ by small margins.
- **Passages are shorter than the 2048-token context.** Most items run 300–450
  tokens, so they measure prediction within a short context rather than
  long-range use of a full window. Longer items should be added.
- **Single author, single register.** Everything here is expository technical
  English written in one voice. It does not cover dialogue, narrative,
  tabular data, or noisy real-world web text, all of which the model will meet.
- **No task-based evaluation.** These measure next-token prediction only. They
  do not test reasoning, arithmetic, or instruction following, which need a
  different artifact with expected outputs rather than raw passages.
