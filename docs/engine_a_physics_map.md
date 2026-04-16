# Engine A Physics Map

Engine A uses the first seven measured boundary dimensions: the per-token
compute/sharpness channel. These dimensions answer one question:

> How much bulk reconstruction does this token still need?

The dimension is real; the axis labels are provisional. Treat this map as a
coordinate chart, not a law of nature.

| Dim | Measured anchors | Engine A meaning | Physics lens | Known solution / model | Test implication |
|---:|---|---|---|---|---|
| 1 | `layer_7`, `layer_1`, `layer_5` | Layer dynamics | Black-hole radial depth | Ryu-Takayanagi minimal surface / quasinormal ringdown | If the state stabilizes by mid-layer, exit early. If it is still ringing, run full depth. |
| 2 | `layer_9`, `hnorm_0`, `cluster_0` | State location | Electron orbital shell | Hydrogen stationary states / eigenshells | Some manifold regions are cheap. Gate thresholds should be cluster-conditioned. |
| 3 | `content_conf`, `logit_gap`, `sup_1` | Distribution sharpness | Wavefunction collapse | Born rule / spectral gap | High top-1 mass and large gap are the first exit signal. |
| 4 | `treuse_2`, `top10_cov`, `sup_1` | Lexical predictability | Resonant standing wave | Bound-state recurrence / cavity modes | Repeated local patterns should exit earlier, but this must not become n-gram overtrust. |
| 5 | `cluster_1`, `layer_5`, `vel_0` | Cluster stability | Laminar boundary layer | Prandtl-Blasius flow | Stable clusters are cheap; turbulent cluster transitions need full depth. |
| 6 | `cluster_1`, `sup_0`, `mom_0` | Superposition clarity | Density-matrix purity | Decoherence / purity | If candidates are cleanly separated, less precision/depth is safe. |
| 7 | `mom_0`, `treuse_2`, `logit_gap` | Distribution shape | Wavepacket moments | Gaussian/coherent-state packet | Skew/kurtosis distinguish true confidence from flat or deceptive confidence. |

## What This Narrows

Engine A should not be a single `softmax > threshold` rule. That is how we loop.
The better gate has three checks:

1. **Sharpness:** top-1 probability, logit gap, entropy, top-k coverage.
2. **Stability:** the same token remains preferred across nearby layers or
   repeated shallow probes.
3. **Geometry:** the hidden state sits in a known laminar cluster or slow-moving
   trajectory region.

The cheap host sequence should therefore compare:

1. Single-confidence gate.
2. Confidence plus logit-gap gate.
3. Confidence plus layer-stability gate.
4. Seven-feature fused gate.
5. Cluster-conditioned thresholds.

## Known Faults

- High confidence can be a false plateau. Require stability or margin.
- Intermediate logits may be uncalibrated. Compare calibration bins before
  trusting thresholds.
- Exit layer can be wrong. Sweep layers before declaring the signal absent.
- Accuracy-only can hide distribution drift. Track KL or top-k agreement after
  exact-match passes.
- Wall-clock speedup is not guaranteed until feature extraction is cheap.
