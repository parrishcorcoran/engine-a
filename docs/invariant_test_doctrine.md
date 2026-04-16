# Engine A Invariant Test Doctrine

The dimension is real. The names of the axes are provisional.

For Engine A, that means we do not worship a specific feature or threshold. We
test invariants of the compute boundary layer.

## The Object

The working claim:

> Transformer depth should be allocated by boundary-state sharpness, not by a
> fixed layer count.

Most tokens are laminar. A smaller number are true branch points. Engine A is
the controller that decides which is which.

## The Five Invariants

| Invariant | What Should Happen | Failure Means |
|---|---|---|
| Calibration monotonicity | Higher gate confidence predicts higher agreement with full depth. | The sensor is not calibrated or the parser is wrong. |
| Entropy-compute inversion | Later/easier laminar tokens need fewer layers at matched fidelity. | The prompt is not becoming deterministic or the simulator/harness is wrong. |
| Layer collapse | Easy tokens stabilize before hard tokens. | Exit layer is wrong, layer tap is wrong, or depth signal is absent. |
| False-plateau guard | Adding margin/stability reduces overconfident wrong exits. | Softmax-only gate is unsafe. |
| Fusion gain | Multi-sensor gate beats any single sensor at the same fidelity. | We are overfitting redundant features or missing the real channel. |

## Local Command

```bash
python measurements/synthetic_engine_a.py --seeds 100
```

Pass condition:

- All invariants pass in the toy world.
- The fused gate improves the false-exit/skip tradeoff over confidence-only.

## Host Translation

| Toy invariant | Real-model metric |
|---|---|
| Calibration monotonicity | Accuracy by confidence bucket is monotonic or mostly monotonic. |
| Entropy-compute inversion | Skip rate rises on easier/later deterministic tokens. |
| Layer collapse | Mid-layer top token matches final top token more often on easy tokens than hard tokens. |
| False-plateau guard | Confidence+gap+stability reduces wrong exits versus confidence alone. |
| Fusion gain | K=7 or K=15-20 feature gate beats one-feature threshold at matched fidelity. |

## Design Rule

Do not optimize for skip rate first. Optimize for calibrated refusal.

The gate earns trust when it says "not yet" on boundary tokens. Once refusal is
correct, we lower compute on the laminar majority.
