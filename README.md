# Engine A

Dynamic boundary-compute experiments for skipping wasteful transformer layers.

## Core Claim

Engine A reads the compute/sharpness side of the boundary layer. As a token
becomes laminar, the model should not need the full depth of the backbone. A
token near a branch point should still receive full compute.

This repo is a clean restart of the Engine A work. It keeps the empirical facts
we already know, but avoids looping by making the branch tree inclusive from
the beginning.

## What We Already Know

- The clean per-sequence local boundary manifold is about 7 dimensions across
  BitNet 2B and Llama 8B.
- The top 7 gate features recover about 80 percent of the achievable compute
  skip signal.
- K=15-20 is the practical deployment feature target.
- K=40-50 is the measured max-accuracy frontier in the old feature sweep.
- Layer-wise trajectory features were the single biggest jump, so Engine A must
  treat depth as a real physical coordinate, not just a software loop.

## Physics Lens

- **Boundary layer:** most tokens are laminar; only branch tokens need full bulk
  reconstruction.
- **Electron cloud:** easy tokens are collapsed wavefunctions; hard tokens are
  genuine superpositions.
- **Black-hole radial depth:** transformer layers behave like a radial bulk
  coordinate. If the boundary readout has stabilized by mid-layer, later layers
  are redundant reconstruction.

## Local Pre-Host Tests

Run the stdlib-only simulator first:

```bash
python measurements/synthetic_engine_a.py --seeds 100
```

Then test the branch tree:

```bash
python measurements/simulate_host_branches.py --mode all
```

Only after those pass should the host run the Hugging Face smoke harness:

```bash
python measurements/hf_engine_a_smoke.py \
  --model /path/to/model \
  --exit_layers 16 \
  --thresholds 0.90 \
  --gates confidence,margin,stability,fused
```

## Success Metrics

- `fidelity`: accepted early-exit tokens match the full-model top token.
- `exit_rate`: fraction of tokens allowed to skip remaining layers.
- `avg_layers_skipped`: expected skipped layers per token.
- `false_exit_rate`: wrong early exits among all tokens.
- `boundary_recall`: hard/boundary tokens correctly sent to full depth.
- `skip_gain`: exit rate at the target fidelity threshold.

The target behavior is not maximum skipping. The target is calibrated skipping:
laminar tokens exit early; boundary tokens do not.

## Repo Map

- `docs/engine_a_physics_map.md`: the 7D Engine A map.
- `docs/invariant_test_doctrine.md`: invariants that prevent looping.
- `docs/inclusive_branch_tree.md`: pass/fail routing for host runs.
- `docs/model_targets.md`: Qwen/Qwen-like host model guidance.
- `docs/engine_ab_coupling.md`: why Engine A may need Engine B veto signals.
- `docs/host_runbook.md`: detailed host execution sequence.
- `docs/result_schema.md`: standard result format and labels.
- `docs/test_ladder.md`: tests from cheapest to hardest.
- `docs/gemini_cli_prompt.md`: handoff prompt for the Z8.
- `measurements/synthetic_engine_a.py`: stdlib-only toy simulator.
- `measurements/free_signal_budget.py`: cheapest/free signal budget simulator.
- `measurements/simulate_host_branches.py`: branch tree fixture/grid runner.
- `measurements/hf_engine_a_smoke.py`: real-model logical early-exit harness.
