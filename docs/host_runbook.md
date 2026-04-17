# Host Runbook

This is the detailed run sequence for the HP Z8 G4 or any high-RAM host.

The rule: do not optimize wall-clock until logical early exit is calibrated.

If the host has hundreds of GB per CPU socket, also read
`docs/high_ram_host_profile.md` before launching real-model sweeps. RAM capacity
is not the bottleneck; NUMA traffic, bandwidth, and full-vocabulary logits are.

## 0. Local Preflight

Run from the repo root:

```bash
python measurements/synthetic_engine_a.py --seeds 100
python measurements/free_signal_budget.py --seeds 100
python measurements/simulate_host_branches.py --mode all
python measurements/physics_monte_carlo.py --models qwen3,qwen2 --engine_b_veto
python -m py_compile measurements/*.py
```

Pass:

- All five synthetic invariants pass.
- The free signal budget confirms Engine B veto reduces false exits.
- Branch fixtures pass.
- Branch grid has no `ambiguous` bucket.
- The Monte Carlo ranks at least one Qwen-class full-precision run as fitting.
- Python files compile.

Fail:

- Fix the local harness first. Do not load a model.

## 1. Minimal Host Smoke

Start with one layer and one threshold:

```bash
python measurements/hf_engine_a_smoke.py \
  --model /path/to/model \
  --exit_layers 16 \
  --thresholds 0.90 \
  --gates confidence,margin,stability,fused \
  --max_prompts 8
```

For Qwen3-8B specifically, start at layer 18 because the model card lists 36
layers:

```bash
python measurements/hf_engine_a_smoke.py \
  --model Qwen/Qwen3-8B \
  --device cpu \
  --dtype float32 \
  --exit_layers 18 \
  --thresholds 0.90 \
  --gates confidence,margin,stability,fused \
  --eval_tail_tokens 128 \
  --max_prompts 8
```

Read the output by gate:

- `confidence` is the naive baseline.
- `margin` adds logit-gap protection.
- `stability` adds previous-layer agreement and entropy protection.
- `fused` combines sharpness, margin, entropy, top-k coverage, and stability.

Pass:

- At least one guarded/fused gate has useful `exit_rate`.
- `fidelity` improves over confidence-only or false exits fall materially.
- Calibration buckets are broadly monotonic.

Fail:

- Route through `docs/inclusive_branch_tree.md`.

## 2. Layer Sweep

```bash
python measurements/hf_engine_a_smoke.py \
  --model /path/to/model \
  --exit_layers 8,12,16,20,24 \
  --thresholds 0.90 \
  --gates confidence,margin,stability,fused \
  --output_csv outputs/layer_sweep.csv
```

Look for:

- The earliest layer with tolerable fidelity.
- The knee where `avg_layers_skipped` is still meaningful.
- Whether calibration becomes monotonic only at later layers.

Routing:

- Good only at late layers: `late-only signal` or `shallow signal`.
- No layer works: `layer signal absent` after checking layer taps.
- Earlier layers confidently wrong: `false plateau`.

## 3. Threshold Sweep

```bash
python measurements/hf_engine_a_smoke.py \
  --model /path/to/model \
  --exit_layers 12,16,20 \
  --thresholds 0.80,0.85,0.90,0.95,0.98 \
  --gates confidence,margin,stability,fused \
  --output_csv outputs/threshold_sweep.csv
```

Look for:

- Matched-fidelity frontier: highest `exit_rate` at a chosen fidelity target.
- Whether guarded/fused gates dominate confidence-only.
- Whether strict thresholds produce `laminar under-skip`.

Routing:

- High fidelity, near-zero exit: `threshold too strict`.
- High exit, low fidelity: `threshold too loose` or `boundary blindness`.
- Good logical frontier: `first-order signal`.

## 4. Prompt-Class Sweep

Make a prompt file with one prompt per line:

```text
The capital of France is
The function returns the value of
In JSON, string values are enclosed in
The secret password is Supernova. Based on the text, the secret password is
```

Run:

```bash
python measurements/hf_engine_a_smoke.py \
  --model /path/to/model \
  --prompt_file prompts/easy_factual.txt \
  --exit_layers 12,16,20 \
  --thresholds 0.90,0.95 \
  --gates confidence,margin,stability,fused \
  --output_csv outputs/easy_factual.csv
```

Repeat for:

- Easy factual.
- Code.
- Long-context retrieval.
- Creative/open-ended.
- Adversarial false-plateau prompts.

Routing:

- Works on easy factual only: Engine A signal exists but is domain-limited.
- Works on code/structured text: strong laminar-token signal.
- Fails on creative/open-ended: expected; treat as boundary-heavy regime.

## 5. Long-Context Drift

Use prompt files at increasing context lengths.

```bash
python measurements/hf_engine_a_smoke.py \
  --model /path/to/model \
  --prompt_file prompts/context_2048.txt \
  --exit_layers 16,20,24 \
  --thresholds 0.90,0.95 \
  --gates fused \
  --output_csv outputs/context_2048.csv
```

Repeat for 512, 2048, 4096, and 8192 tokens if supported.

Routing:

- Fidelity drops with context length: `context drift`.
- Top-1 stays but top-k overlap drops: `distribution drift`.
- Threshold needs to move with context: use context-band calibration.

## 6. Systems Gate

Only start systems work when:

- Logical `fidelity` is acceptable.
- `exit_rate` is non-trivial.
- `avg_layers_skipped` is meaningful.
- False plateau cases are blocked by guarded/fused gates.

Then:

1. Implement cheap feature extraction in C++.
2. Keep full-depth fallback.
3. Measure tokens/sec and tokens/joule.
4. Compare against full-depth baseline at matched quality.

If logical skip works but wall-clock does not, route to `feature overhead`.

## Suggested Output Files

Use `outputs/` locally. It is gitignored.

- `outputs/layer_sweep.csv`
- `outputs/threshold_sweep.csv`
- `outputs/context_512.csv`
- `outputs/context_2048.csv`
- `outputs/context_4096.csv`
- `outputs/context_8192.csv`
- `outputs/result_log.md`

## Stop Conditions

Stop and reassess if:

- Full-depth baseline is unstable.
- Calibration buckets are inverted at every tested layer.
- Guarded/fused gates do not improve false exits.
- Long-context drift destroys fidelity and cannot be band-calibrated.
- Wall-clock overhead exceeds saved layer compute after C++ feature extraction.
