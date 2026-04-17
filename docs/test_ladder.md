# Engine A Test Ladder

Start with cheap simulations. Move to real models only when the branch tree and
toy invariants are clean.

## Tier 0: Local Simulators

Runtime: seconds.

```bash
python measurements/synthetic_engine_a.py --seeds 100
python measurements/free_signal_budget.py --seeds 100
python measurements/simulate_host_branches.py --mode all
python measurements/physics_monte_carlo.py --models qwen3,qwen2 --engine_b_veto
```

Pass:

- All Engine A invariants pass.
- The cheap/free signal budget passes.
- Branch fixtures pass.
- Grid has no `ambiguous` bucket.
- The model/runtime Monte Carlo gives a plausible first host target.

Fail:

- Fix the harness or branch tree before using a host model.

## Tier 1: Real-Model Logical Early Exit

Runtime: minutes.

```bash
python measurements/hf_engine_a_smoke.py \
  --model /path/to/model \
  --exit_layers 16 \
  --thresholds 0.90 \
  --gates confidence,margin,stability,fused \
  --max_tokens 512
```

Pass:

- Confidence buckets are monotonic.
- Early-exit candidates match final top token at high fidelity.
- `avg_layers_skipped` is non-zero.

Fail:

- If all buckets are flat, the layer tap or lm_head projection is wrong.
- If high-confidence exits are wrong, add logit-gap and stability gates.
- If no tokens exit, lower threshold or try a later exit layer.

## Tier 2: Exit-Layer Sweep

Runtime: minutes to tens of minutes.

```bash
for layer in 8 12 16 20 24; do
  python measurements/hf_engine_a_smoke.py \
    --model /path/to/model \
    --exit_layers "$layer" \
    --thresholds 0.90 \
    --gates confidence,margin,stability,fused \
    --max_tokens 512
done
```

Pass:

- There is a clear knee where fidelity rises and skip remains useful.

Fail:

- If earlier and later layers are identical, check hidden-state capture.
- If only very late layers work, Engine A still exists but speedup ceiling is
  lower for this model.

## Tier 3: Gate Variants

Compare:

1. Confidence only.
2. Confidence plus logit gap.
3. Confidence plus layer stability.
4. Confidence plus local hidden-state stability.
5. Seven-feature fused gate.

Pass:

- Fused or guarded gates reduce false exits at matched skip.

Fail:

- If confidence-only wins, feature set is redundant.
- If every gate fails, move to trained per-layer heads.

## Tier 4: Long-Context Drift

Run the same harness at 512, 2048, 4096, and 8192 tokens if the model supports
it.

Pass:

- Calibration remains stable or can be corrected by context-length bands.

Fail:

- If threshold shifts with length, use per-context calibration.

## Tier 5: Wall-Clock Engine A

Only after logical early exit works:

1. Move feature extraction into C++.
2. Preserve a full-depth fallback.
3. Measure tokens/sec and tokens/joule.
4. Report quality at matched fidelity thresholds.

Do not optimize wall-clock before the gate refuses boundary tokens correctly.
