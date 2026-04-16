# Gemini CLI Handoff Prompt

Use this prompt on the HP Z8 G4.

```text
You are working in the `engine-a` repo.

Goal:
Test whether Engine A can skip transformer depth on laminar tokens while
refusing to exit on boundary tokens. Do not optimize for skip rate first.
Optimize calibrated refusal first.

Read:
- README.md
- docs/engine_a_physics_map.md
- docs/invariant_test_doctrine.md
- docs/inclusive_branch_tree.md
- docs/host_runbook.md
- docs/result_schema.md
- docs/test_ladder.md
- docs/fault_model.md
- measurements/hf_engine_a_smoke.py

Work style:
- Start with the fastest tests.
- Run one variable at a time.
- Route every result through the inclusive branch tree.
- If a test misses, label the failure before changing code.
- Do not move to C++ until logical early exit has a strong signal.

Before host-model tests:
- Run `python measurements/synthetic_engine_a.py --seeds 100`.
- Run `python measurements/simulate_host_branches.py --mode all`.
- Confirm all invariants pass and the branch grid has no ambiguous bucket.

First host tests:
1. Run the HF smoke harness at layer 16, threshold 0.90 with all gates.
2. Sweep exit layers: 8, 12, 16, 20, 24.
3. Sweep thresholds: 0.80, 0.85, 0.90, 0.95, 0.98.
4. Compare confidence-only, margin, stability, and fused gates.
5. Only then test long-context drift.

Metrics to watch:
- fidelity
- exit_rate
- false_exit_rate
- boundary_recall
- avg_layers_skipped
- calibration buckets
- distribution drift / top-k agreement
- wall-clock only after logical signal passes

Interpretation:
- If confidence buckets are not monotonic, fix layer tap/norm or try later layer.
- If high-confidence exits are wrong, this is false plateau; add stability.
- If fidelity is high but exits are near zero, threshold is too strict.
- If logical skip works but wall-clock fails, feature extraction is the problem,
  not the Engine A signal.
```
