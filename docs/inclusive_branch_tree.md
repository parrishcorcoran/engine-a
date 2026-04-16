# Inclusive Branch Tree

Use this after every Engine A run. Every outcome should route to a label and a
next action.

## Start

Run:

```bash
python measurements/synthetic_engine_a.py --seeds 100
python measurements/simulate_host_branches.py --mode all
```

If either fails, fix the local harness first.

## Tier 0: Baseline

### Baseline Fails

Symptoms:

- Full-depth model gives inconsistent or wrong answers.

Diagnosis:

- `parser/eval failure` if output is acceptable but marked wrong.
- `baseline instability` if full-depth output changes or misses the task.

Next:

- Fix prompt, parser, temperature, or model before testing Engine A.

## Tier 1: Calibration

### Confidence Is Not Predictive

Symptoms:

- High confidence bucket is not more accurate than low confidence bucket.

Diagnosis:

- `no calibration`.

Next:

- Try later exit layer.
- Apply final norm before `lm_head`.
- Add logit gap and entropy features.

### High Confidence Is Wrong

Symptoms:

- High exit rate but bad fidelity.
- Wrong exits cluster in high-confidence buckets.

Diagnosis:

- `false plateau` if intermediate layer is confident before later correction.
- `threshold too loose` if all weak candidates are accepted.
- `boundary blindness` if hard tokens exit early.

Next:

- Add stability guard.
- Raise threshold.
- Add hard-token detector.

## Tier 2: Utility

### Safe But Useless

Symptoms:

- Fidelity is high.
- Exit rate is near zero.

Diagnosis:

- `threshold too strict`.
- `laminar under-skip`.

Next:

- Lower threshold.
- Add cluster-conditioned thresholds.
- Try later layer if current layer is too early.

### Useful Logical Skip

Symptoms:

- Fidelity is high.
- Exit rate is non-trivial.
- Average layers skipped is meaningful.

Diagnosis:

- `first-order signal`.

Next:

- Sweep exit layers and thresholds.
- Add guarded/fused gates.
- Move to long-context drift.

## Tier 3: Layer Sweep

### No Layer Works

Symptoms:

- Every exit layer has bad fidelity or zero exits.

Diagnosis:

- `layer signal absent` if capture is correct.
- `layer tap mismatch` if results look impossible.

Next:

- Verify hidden-state indexing and normalization.
- If correct, move to trained per-layer classifier.

### Only Late Layers Work

Symptoms:

- Layers near the end work, mid-layer does not.

Diagnosis:

- `late-only signal`.

Next:

- Speedup ceiling is lower; test whether skip is still worth C++ work.

## Tier 4: Distribution And Context

### Exact Match Passes But Distribution Drifts

Symptoms:

- Top-1 agreement high.
- KL/top-k agreement poor.

Diagnosis:

- `distribution drift`.

Next:

- Add KL/top-k guard or increase threshold.

### Short Context Works, Long Context Fails

Symptoms:

- Calibration changes with context length.

Diagnosis:

- `context drift`.

Next:

- Calibrate thresholds per context band.

## Tier 5: Systems

### Logical Skip Works But Wall-Clock Does Not

Symptoms:

- Logical layers skipped are high.
- Tokens/sec or tokens/joule does not improve.

Diagnosis:

- `feature overhead`.

Next:

- Move features into C++.
- Drop expensive features.
- Measure tokens/joule over long runs.

## Failure Taxonomy

| Label | Meaning | Next Move |
|---|---|---|
| parser/eval failure | Eval marked acceptable output wrong | Fix parser |
| baseline instability | Full-depth baseline is not stable | Fix prompt/model |
| no calibration | Confidence does not predict correctness | Try later layer/norm/gap |
| false plateau | Intermediate layer is confidently wrong | Add stability guard |
| threshold too strict | Safe but exits almost nothing | Lower threshold/adapt by cluster |
| threshold too loose | Exits too much and loses fidelity | Raise threshold/add hard-token guard |
| boundary blindness | Hard tokens exit early | Add entropy/moment/velocity features |
| laminar under-skip | Easy tokens still run full depth | Cluster-conditioned thresholds |
| layer tap mismatch | Hidden-state capture/projection is wrong | Verify layer index and norm |
| layer signal absent | No layer carries usable early signal | Try trained classifier |
| late-only signal | Only late exit layers work | Measure whether speedup ceiling is enough |
| distribution drift | Exact match hides logit drift | Add KL/top-k guard |
| context drift | Gate shifts with context length | Context-band calibration |
| feature overhead | Logical speedup loses wall-clock | C++ features/tokens-joule |
| shallow signal | Gate is safe but saves little depth | Try earlier exit layers or accept lower ceiling |
| first-order signal | Gate skips safely | Sweep and move deeper |
| partial unsafe signal | Gate has coverage but fidelity is not deployable | Add guards or raise threshold |
| weak signal | Some structure but little utility | Try later layers or fused features |

## Result Log Template

```text
Run:
Command:
Model:
Exit layer:
Threshold:

Metrics:
- baseline_ok:
- calibration_monotonic:
- fidelity:
- exit_rate:
- false_exit_rate:
- boundary_recall:
- avg_layers_skipped:
- distribution_drift:
- context_length:
- wall_clock_delta:

Diagnosis label:

Interpretation:

Next single change:
```
