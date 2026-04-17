# High-RAM Host Profile

This profile assumes a dual-socket host with roughly 350 GB or more RAM attached
to each CPU socket.

## What This Changes

RAM capacity is not the bottleneck for Qwen-class 7B-9B models. The bottlenecks
become:

- CPU memory bandwidth.
- Cross-socket NUMA traffic.
- Python and `transformers` overhead.
- Full-vocabulary logits over long contexts.

Use the RAM to preserve observability first. Do not start by quantizing away the
signal unless full-precision runs are impossible.

## Approximate Qwen3-8B Memory Budget

For 8.19B parameters:

| Mode | Weight memory |
| --- | ---: |
| fp32 | about 33 GB |
| bf16/fp16 | about 17 GB |
| int8 | about 9 GB |
| 4-bit | about 5 GB plus overhead |

The KV cache is also comfortable at this scale. With 36 layers, 8 KV heads, and
128-dimensional KV heads, bf16 KV is roughly 144 KiB per token per sequence:

| Context | bf16 KV | fp32 KV |
| --- | ---: | ---: |
| 2K | about 288 MB | about 576 MB |
| 32K | about 4.5 GB | about 9 GB |
| 128K | about 18 GB | about 36 GB |

The larger hidden cost for Engine A experiments is often not the KV cache. It is
asking `transformers` for all hidden states and full-vocabulary logits over many
positions. Use `--eval_tail_tokens` to score only the token positions that matter
for the current test.

## Recommended Precision Order

1. Use `float32` on CPU for first signal discovery if runtime is tolerable.
2. Use `bfloat16` only if the CPUs support it well.
3. Use quantized models for throughput checks after the signal is understood.

Quantization is useful, but it can blur exactly the logit-gap and entropy
geometry Engine A is trying to measure.

## NUMA Rule

Keep each model process inside one socket unless you are deliberately testing
cross-socket behavior.

On Linux, prefer one sweep per socket:

```bash
numactl --cpunodebind=0 --membind=0 python measurements/hf_engine_a_smoke.py ...
numactl --cpunodebind=1 --membind=1 python measurements/hf_engine_a_smoke.py ...
```

Do not split one small 8B model across both sockets just because memory exists.
That often trades capacity for cross-socket latency.

## First Qwen3-8B CPU Run

```bash
python measurements/hf_engine_a_smoke.py \
  --model Qwen/Qwen3-8B \
  --device cpu \
  --dtype float32 \
  --exit_layers 18 \
  --thresholds 0.90 \
  --gates confidence,margin,stability,fused \
  --max_tokens 1024 \
  --eval_tail_tokens 128 \
  --max_prompts 8
```

If this is clean, widen one axis at a time:

```bash
python measurements/hf_engine_a_smoke.py \
  --model Qwen/Qwen3-8B \
  --device cpu \
  --dtype float32 \
  --exit_layers 9,12,15,18,21,24,27,30 \
  --thresholds 0.85,0.90,0.95 \
  --gates confidence,margin,stability,fused \
  --max_tokens 2048 \
  --eval_tail_tokens 256 \
  --output_csv outputs/qwen3_8b_cpu_sweep.csv
```

## Long-Context Rule

For long-context Engine A runs, do not evaluate every filler token unless that is
the actual question. Usually we care about:

- The answer region.
- The last 128-512 positions.
- Positions around boundary transitions.
- Needle retrieval answer tokens.

Run the long context with a small `--eval_tail_tokens` first. Only expand to all
positions if the tail signal passes.

## Best Use Of 700 GB Total RAM

Use the host as a signal laboratory:

- Run full-precision Qwen3-8B on socket 0.
- Run an alternate layer/threshold sweep on socket 1.
- Keep raw JSONL/CSV outputs for branch-tree comparison.
- Avoid wall-clock optimization until logical skip fidelity passes.

The goal is not to prove the host can hold the model. It can. The goal is to
find the cheapest reliable boundary signal before spending time on kernels.
