# Changelog

## Data phase closed, model phase built — 2026-08-15

The project moved from "a cleaned corpus exists" to "a trainable 32.7M-parameter
Transformer whose architecture is proven". Three artifacts were validated and
frozen, the dataset builder was rewritten to remove a memory ceiling, and the
complete model was implemented one mathematically-tested component at a time.

Test suite: **557 → 965 tests**, all passing.

---

## 1. Frozen artifacts

Each was validated by recomputing every claim from the underlying bytes, then
stamped with hashes so later drift is detectable rather than assumed absent.

### Pretraining corpus v0.1 — FROZEN

`data/cleaned/pretraining/v0.1/FROZEN.json` · 30/30 checks · 115.5s

```
documents            1,160,605     agreed by 6 independent sources
characters       5,056,851,885
UTF-8 text bytes 5,092,495,573
JSONL bytes      5,533,247,251
shards                      41
evaluation SHA-256   6c994e25…    matches the pinned digest
```

Beyond counting, the validator recomputes the *derived* fields —
`document_id == make_document_id(text_sha256)` and
`split_group == make_split_group(url, document_id)` — for all 1.16M documents,
and confirms every stored text is a fixed point of the normalizer. Uniqueness is
proved without holding the corpus in memory, by exploiting the fact that a
`document_id` collision is exactly a 12-byte digest-prefix collision.

### Tokenized dataset v0.1 — FROZEN

`data/tokenized/v0.1/FROZEN.json` · 31/31 checks · 279.4s

```
                     train      validation
documents        1,148,751          11,854
tokens       1,083,858,263      10,870,996
shards                 109               2
windows of 2048    529,227           5,308
                 ────────────────────────
total tokens         1,094,729,259
```

The two checks that carry the most weight:

- **Document framing was walked, not sampled.** The entire 1.09B-token stream ran
  through a two-state machine requiring BOS/EOS to strictly alternate. BOS count
  came out at exactly 1,148,751 — one per declared document. A single EOS dropped
  at a shard boundary would pass any sampling check and silently teach the model
  that documents do not end.
- **The split re-derives from the frozen corpus.** Recomputed from scratch, both
  ordered document-identity digests reproduced the manifest exactly, which proves
  train/validation document disjointness rather than inferring it from policy.

Also verified: zero leakage of the seven non-boundary special tokens across 5 GB
of web scrape, and all 108 internal shard boundaries read whole and aligned.

---

## 2. Streaming refactor of the dataset builder

`build_pretraining_dataset.py` collected every document into a list before
tokenization, and `dataset_training.py` then materialized the same corpus twice
more; `normalize_text` ran roughly four times over all 5 GB.

Replaced with a three-pass streaming pipeline:

```
pass 1   identity, split_group, normalized byte count   -> split plan
pass 2   train documents      -> tokenize -> train shards
pass 3   validation documents -> tokenize -> validation shards
```

Three passes rather than two because two would require the shard writer to accept
interleaved writes for both splits. Keeping it a single sequential stream writer
is what makes the output bytes provably unchanged. Every document is still
tokenized exactly once.

Peak memory is now set by the number of distinct identifiers, not by the volume
of text. One-shot iterators are rejected with a clear error rather than quietly
buffered, since buffering is the bug being removed.

**Proof of equivalence:** `tests/data/test_dataset_training_streaming_regression.py`
embeds the pre-refactor implementation verbatim as a reference oracle and asserts
SHA-256 equality of every produced file — the `.bin` shards, both split manifests,
and `dataset_manifest.json` — across four geometries. 28 tests.

A cross-pass stability guard was added: an ordered document-identity digest
computed in pass 1 and recomputed in pass 2 catches a source that changes between
passes, including a swap that preserves the document count.

---

## 3. The model

`src/llm/model/` is new. Built in dependency order, each component with
mathematical reference tests before the next was added.

| Module | Tests | Parameters |
|---|---:|---:|
| `config.py` | 26 | — |
| `rmsnorm.py` | 50 | 512 per norm |
| `rope.py` | 79 | 0 |
| `attention.py` | 64 | 1,048,576 |
| `feedforward.py` | 44 | 2,359,296 |
| `block.py` | 49 | 3,408,896 |
| `transformer.py` | 58 | **32,741,888** |

```
embedding      24,000 × 512                 12,288,000
6 × block                                   20,453,376
final RMSNorm                                      512
tied output projection                               0
                                            ──────────
                                            32,741,888
```

Asserted at runtime, and every intermediate total checked against a real module
rather than the config's own arithmetic.

### Frozen architecture decisions

```
vocab_size       24,000        rope_theta          10,000
context_length    2,048        rope pairing        interleaved
n_layers              6        rope tables         precomputed, FP32,
hidden_size         512                            non-persistent
n_heads               8        norm placement      pre-norm
head_dim             64        attention masking   SDPA is_causal=True
ffn_hidden_size   1,536        KV cache            none (v0.1)
rms_norm_eps       1e-6        init                N(0, 0.02), norms at 1.0
tie_embeddings     True        biases              none
```

### Notes on specific choices

**Weight tying is structural.** The output projection is
`F.linear(hidden, token_embedding.weight)`; there is no second matrix to drift.
Tests assert `data_ptr` identity, that exactly one `(24000, 512)` parameter
exists, and that vocabulary rows absent from a batch still receive gradient —
which can only arrive through the projection role.

**RoPE tables resist downcast.** `model.to(torch.bfloat16)` would silently
degrade the position signal, and casting back cannot recover the bits, so
`_apply` rebuilds the tables from their deterministic definition instead.

**Causality is re-proved at every level of composition** — attention, block, and
whole model — in both directions. The backward form asserts that the gradient of
an earlier output leaves later inputs at exactly zero.

**Residual wiring is proved by silencing branches.** With `attention.o_proj` and
`feedforward.down_proj` zeroed, both branches emit exactly zero and the block
must be the identity. A missing residual returns zeros instead.

**Initialization is one model-wide policy** (`Transformer.reset_parameters`), not
per-module defaults. No depth-dependent residual scaling yet — the first thing to
try if a deeper configuration misbehaves.

---

## 4. Tiny-overfit proof — PASS

Full 32.7M model, 8 × 64 random token ids, 200 steps, AdamW lr=1e-3:

```
step   0     loss 10.1981     grad norm 4.176
step  40     loss  0.0140     grad norm 0.038
step 199     loss  0.0002

token accuracy 100.00%       reduction 57,945×
```

Random ids rather than text, deliberately: random sequences have no structure to
generalize from, so the only route to zero loss is genuine memorization.

Initial loss matches theory. With `σ = √512 × 0.02 = 0.4525`,
`E[CE] ≈ ln V + σ²/2 = 10.188` against 10.1981 observed — random logits cost
slightly *more* than uniform, because the target logit is as likely to be low as
high while `logsumexp` rises with spread.

**Measured throughput: 612 tokens/s on CPU.** Against the 1,083,858,263-token
training split that is **20.5 days per epoch**, so full pretraining needs a CUDA
GPU. The installed `torch 2.13.0+cpu` has no CUDA support compiled in.

---

## File inventory

### New — 21 files, 266 KB

**Validation and experiment scripts**

| File | Size |
|---|---:|
| `scripts/validate_pretraining_corpus.py` | 49.0 KB |
| `scripts/validate_tokenized_dataset.py` | 37.8 KB |
| `scripts/tiny_overfit.py` | 8.3 KB |

**Model package** (`src/llm/model/`)

| File | Size |
|---|---:|
| `__init__.py` | 0.8 KB |
| `config.py` | 10.4 KB |
| `rmsnorm.py` | 4.6 KB |
| `rope.py` | 10.2 KB |
| `attention.py` | 6.4 KB |
| `feedforward.py` | 4.1 KB |
| `block.py` | 3.9 KB |
| `transformer.py` | 8.6 KB |

**Tests** — 408 new tests

| File | Size | Tests |
|---|---:|---:|
| `tests/data/test_dataset_training_streaming_regression.py` | 18.2 KB | 28 |
| `tests/model/test_config.py` | 6.9 KB | 26 |
| `tests/model/test_rmsnorm.py` | 11.6 KB | 50 |
| `tests/model/test_rope.py` | 18.2 KB | 79 |
| `tests/model/test_attention.py` | 18.0 KB | 64 |
| `tests/model/test_feedforward.py` | 12.4 KB | 44 |
| `tests/model/test_block.py` | 14.1 KB | 49 |
| `tests/model/test_transformer.py` | 16.3 KB | 58 |
| `tests/model/test_overfit.py` | 5.3 KB | 10 |

**Configuration**

- `.gitignore` — 1.0 KB, excludes bulk payloads by extension rather than by
  directory, so manifests and freeze stamps stay in history alongside the data
  they describe. Tracks `artifacts/tokenizer-E011/` only.

### Modified — 2 files

- `src/llm/data/dataset_training.py` — three-pass streaming rewrite
- `scripts/build_pretraining_dataset.py` — document factory instead of `list(...)`,
  plus a train/validation summary report

### Generated by the validators — 5 files

- `data/cleaned/pretraining/v0.1/FROZEN.json` (13.1 KB, per-shard SHA-256 for 41 shards)
- `data/audits/corpus-v0.1-validation.json` (18.0 KB)
- `data/tokenized/v0.1/FROZEN.json` (1.7 KB)
- `data/audits/dataset-v0.1-validation.json` (8.2 KB)
- `data/audits/tiny-overfit-v0.1.json` (full 200-step loss history)

### Deleted

- `data/language_sanity.py` — obsolete draft superseded by
  `src/llm/data/language_sanity.py`; used `ScriptSanityVerdict` where the package
  uses `ScriptVerdict`, and nothing imported it
- All `__pycache__/`, `.pytest_cache/`, `src/my_llm.egg-info/` (1.19 MB)

---

## Open items

**Git is not installed on this machine.** Everything above is unversioned,
including three freeze stamps whose purpose is to make tampering detectable.
`.gitignore` is written and waiting. Simulated tracking footprint: ~2,411 files
and 96.2 MB, of which 44.7 MB is `data/tokenizer_sources/` (not reproducible —
Wikipedia articles change) and 40.3 MB is `data/tokenizer_training/` (rebuildable
from those sources, and a candidate for exclusion).

**Duplicate audit index.** `near-duplicates-100k-v02.sqlite` and
`near-duplicates-100k-v2.sqlite` are byte-identical (SHA-256 `68C516BA…`); one is
715 MB of reclaimable disk. Their two JSON reports are **not** identical and both
should be kept. Git offers no protection here — `data/audits/` is gitignored — so
this decision is independent of version control.

**Hardware.** The training engine's design depends on the target device: FP32
versus BF16 as the default path, whether gradient accumulation is needed, and
whether checkpointing must survive multi-day or multi-hour runs.

---

## Next

```
7.  Training engine      src/llm/training/{trainer,checkpoint,schedule}.py
8.  Small real-data run  hundreds of steps on real shards
9.  Full pretraining     1,083,858,263 tokens
10. Generation           greedy, temperature, top-k, EOS stopping
11. Evaluation           validation loss, frozen eval set, free generation
```
