# Cloud GPU training

Setup for pretraining `base-v0.1` on a rented NVIDIA GPU. Written for RunPod,
but nothing here is provider-specific beyond the transfer commands.

The local machine (Ryzen 7 7730U, AMD integrated graphics) has no CUDA path.
Measured CPU throughput is ~680–800 tokens/s, which is ~18 days for one epoch
over the 1.084B-token training split. A single consumer GPU brings that into
the hours range.

## What to transfer

| Path | Size | Needed for |
|---|---:|---|
| the repository | ~1 MB | code |
| `data/tokenized/v0.1/` | ~2.2 GB | training |
| `artifacts/tokenizer-E011/` | 6.3 MB | generation, later |

The 1.9 GB cleaned corpus is **not** needed. Training reads only the `.bin`
token shards.

## 1. Pick a pod

A 32.7M-parameter model is small; the constraint is CUDA, not capacity.
An RTX 4090 (24 GB) is sufficient. An RTX 5090 (32 GB) gives more headroom for
the batch-size sweep, which is the one place extra VRAM pays off here.

Choose a PyTorch or CUDA base image if offered — it saves the driver setup.

## 2. Install

Order matters. Install the CUDA build of torch **first**:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

With torch already satisfied, `pip install -e .` leaves it alone. Reversed, pip
pulls the CPU-only wheel from PyPI and the failure surfaces much later as
`Torch not compiled with CUDA enabled`.

Verify before going further:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
nvidia-smi
```

## 3. Transfer the data

```bash
# from the local machine
scp -P <port> -r data/tokenized/v0.1 root@<host>:/workspace/my_LLM-main/data/tokenized/
scp -P <port> -r artifacts/tokenizer-E011 root@<host>:/workspace/my_LLM-main/artifacts/
```

Then **verify the transfer** rather than assuming it arrived intact:

```bash
python scripts/validate_tokenized_dataset.py \
    --dataset data/tokenized/v0.1 \
    --tokenizer artifacts/tokenizer-E011/tokenizer.json
```

This re-checks every shard SHA-256 against `FROZEN.json` and re-walks the
BOS/EOS framing. A truncated or corrupted transfer is otherwise invisible until
the loss curve looks subtly wrong.

## 4. Confirm the code works there

```bash
python -m pytest tests -q
```

1,038 tests, a couple of minutes. Catches an environment problem before it
costs GPU-hours.

## 5. Benchmark before committing

```bash
python scripts/benchmark_training.py \
    --dataset data/tokenized/v0.1 \
    --device cuda \
    --precision bf16 \
    --report data/audits/benchmark-<gpu>.json
```

Phase 1 doubles the micro-batch until it stops fitting or stops getting faster,
reporting tokens/s and peak VRAM at each size. Out-of-memory is the expected
terminating condition, not an error.

Phase 2 trains a freshly initialized model for a short burst at each candidate
learning rate from an identical seed, reporting loss reduction and the largest
gradient norm. Prefer a rate that descends well *without* gradient spikes — the
spike is what produces a NaN hours into a run.

It ends by printing a ready-to-paste `train_model.py` command with the measured
micro-batch, the accumulation needed to reach the target effective batch, the
steps for one epoch, and the recommended learning rate.

## 6. Train

Run the command the benchmark printed. Sanity-check two things in it first:

- **Steps.** One epoch over 1,083,858,263 tokens at 65,536 tokens/step is
  **16,538 steps**. Chinchilla-optimal for 32.7M parameters is ~655M tokens, so
  one epoch is already 1.66× that. More epochs mostly buy memorization of
  repeated data — decide deliberately rather than by default.
- **Checkpoint interval.** The default is every 500 steps. On a rented pod that
  can be reclaimed, more frequent is cheaper than the alternative.

Resume is automatic: rerunning the identical command loads the highest-step
checkpoint from `--checkpoint-dir` and continues. `--no-resume` starts over.

## 7. Retrieve the checkpoint

```bash
scp -P <port> root@<host>:/workspace/my_LLM-main/checkpoints/base-v0.1/step-*.pt ./checkpoints/base-v0.1/
```

A checkpoint is ~390 MB: 32.7M parameters plus AdamW's two moment estimates,
all in fp32. Pull it before releasing the pod.

## Notes

**BF16, not FP16.** BF16 keeps FP32's exponent range and needs no gradient
scaler. The `fp16` option exists but has no scaler wired up, and the script
warns if you select it.

**Precision does not change the parameters.** Weights and optimizer state stay
FP32 in both modes; only the forward and backward math is reduced.

**The dataset identity is recorded in every checkpoint.** Resuming against a
different corpus is refused rather than silently accepted, so a re-transferred
or regenerated dataset will be caught.
