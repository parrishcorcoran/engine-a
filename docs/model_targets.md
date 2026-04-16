# Model Targets

Engine A wants overparameterized dense models first. If the model is too small
or already aggressively compressed, there may be less redundant depth to skip.

## Recommended First Target

Use Qwen3-8B or a close 7B-9B dense causal LM.

As of 2026-04-16, the Hugging Face model card for `Qwen/Qwen3-8B` describes it
as:

- 8.2B total parameters.
- 6.95B non-embedding parameters.
- 36 layers.
- GQA with 32 query heads and 8 KV heads.
- 32,768 native context length, with 131,072 tokens via YaRN.
- Supported in recent `transformers`; older versions may not know the `qwen3`
  architecture.

That is a good Engine A test shape: enough layers to expose depth redundancy,
enough overparameterization to expect a laminar/boundary split, and enough
context to test drift.

Primary model-card URL:

```text
https://huggingface.co/Qwen/Qwen3-8B
```

## Suggested Host Commands

Minimal smoke:

```bash
python measurements/hf_engine_a_smoke.py \
  --model Qwen/Qwen3-8B \
  --exit_layers 18 \
  --thresholds 0.90 \
  --gates confidence,margin,stability,fused \
  --max_prompts 8
```

Layer sweep for a 36-layer model:

```bash
python measurements/hf_engine_a_smoke.py \
  --model Qwen/Qwen3-8B \
  --exit_layers 9,12,15,18,21,24,27,30 \
  --thresholds 0.90,0.95 \
  --gates confidence,margin,stability,fused \
  --output_csv outputs/qwen3_8b_layer_sweep.csv
```

Threshold sweep:

```bash
python measurements/hf_engine_a_smoke.py \
  --model Qwen/Qwen3-8B \
  --exit_layers 15,18,21,24 \
  --thresholds 0.80,0.85,0.90,0.95,0.98 \
  --gates confidence,margin,stability,fused \
  --output_csv outputs/qwen3_8b_threshold_sweep.csv
```

## Similar Targets

Use these if Qwen3-8B is not available or if the host has a quantized local
copy:

- Qwen 7B-9B dense variants.
- Llama 3/3.1 8B dense variants.
- Mistral 7B dense variants.
- Gemma-class dense models around 7B-12B.

Avoid starting with:

- Tiny models below about 2B parameters, unless debugging the harness.
- MoE models, because expert routing adds another compute allocation variable.
- Vision-language models, because token/text-only assumptions can break.
- Heavily distilled models, because depth redundancy may already be reduced.

## Why Qwen-Like Models Are Attractive

The Engine A hypothesis is strongest when:

- The model has many layers.
- The hidden state is overcomplete relative to the decision manifold.
- The next-token cloud often collapses before the final layer.
- The model has enough capacity that simple tokens are not actually hard.

Qwen3-8B has the right shape for this first serious host test.
