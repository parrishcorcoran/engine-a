# Physics-Informed Monte Carlo

Yes, we can run a Monte Carlo to predict which model is most likely to work and
roughly how fast it will run. The important caveat: quantum mechanics does not
literally predict transformer wall-clock speed. Use physics equations as priors,
then calibrate them with one real host measurement.

## What The Simulator Predicts

The simulator ranks model and dtype choices by:

- Fit probability on the host.
- Estimated p50/p90 wall-clock seconds.
- Probability of satisfying an Engine A target.
- Median early-exit fidelity.
- Median useful exit rate.
- False-plateau risk.

Run:

```bash
python measurements/physics_monte_carlo.py \
  --context_tokens 2048 \
  --eval_tail_tokens 128 \
  --prompts 8 \
  --samples 2000 \
  --engine_b_veto
```

For a Qwen-only first pass:

```bash
python measurements/physics_monte_carlo.py \
  --models qwen3,qwen2 \
  --dtypes float32,bfloat16,int8 \
  --context_tokens 2048 \
  --eval_tail_tokens 128 \
  --prompts 8 \
  --samples 2000 \
  --engine_b_veto
```

## Physics Priors Used

### Roofline Wall Clock

Wall clock is modeled as:

```text
time >= max(total_flops / effective_flops, streamed_bytes / memory_bandwidth)
```

This is hardware physics, not metaphor. On the Z8-class host, memory bandwidth
and NUMA placement can dominate after model fit is no longer the problem.

### KV And Activation Memory

KV cache memory is modeled as:

```text
2 * layers * kv_heads * head_dim * tokens * bytes_per_value
```

For Engine A smoke tests, the bigger surprise can be hidden states and logits,
not KV. Asking `transformers` for all hidden states and all final logits can
cost more than expected even when RAM capacity is huge.

### Landau-Zener False Plateaus

False plateau risk uses the Landau-Zener transition shape:

```text
P(flip) ~= exp(-2*pi*gap^2 / drift)
```

Interpretation:

- Big logit gap means stable token identity.
- Fast boundary drift means the token can still flip later.
- Engine B veto lowers the danger by refusing exits during memory turbulence.

This is not literal electron tunneling. It is a useful equation shape for
"confident now, wrong later."

### Spiked Random Matrix Separation

The Engine A signal prior assumes overparameterized models expose a cleaner
low-dimensional boundary when the signal eigenvalue separates from the noise
bulk:

```text
signal-to-noise > separation threshold
```

This is why dense 7B-9B models are better first targets than tiny or heavily
distilled models.

### Localization Versus Diffusion

The trajectory can localize into a deterministic continuation or diffuse into
memory ambiguity. This is the electron-cloud lens:

- Localized state: low entropy, high gap, stable top token.
- Diffuse state: memory tension, distractor pressure, false plateau risk.

Engine A should exit localized states. Engine B should veto diffuse states.

## How To Use The Result

The Monte Carlo is for ordering, not final truth.

1. Run the simulator with the planned context size.
2. Pick the best high-signal full-precision target.
3. Run one real host smoke command.
4. Compare observed wall-clock against predicted p50.
5. Adjust `--compute_tflops` or `--mem_bandwidth_gbs`.
6. Rerun the Monte Carlo with calibrated hardware.

If the model fits and wall-clock is merely slow, do not quantize first. Reduce
`--eval_tail_tokens`, prompt count, or sweep width. Quantize after the signal is
understood.
