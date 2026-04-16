# Engine A Fault Model

Engine A failures should be named, not vaguely debugged. The fastest path is to
identify the failure class and make one targeted change.

| Fault | Symptom | Likely Cause | First Response |
|---|---|---|---|
| Parser/eval failure | Generated text is acceptable but marked wrong | Evaluation too strict | Inspect decoded tokens, relax parser |
| Baseline instability | Full-depth baseline is inconsistent | Prompt/model not deterministic enough | Lower temperature, stronger prompt, more seeds |
| No calibration | Confidence buckets do not predict correctness | Sensor is not aligned with full model | Try later exit layer, add logit gap |
| False plateau | High confidence exits are wrong | Intermediate layer overconfident before correction | Add layer-stability or margin guard |
| Threshold too strict | Fidelity good but exit rate near zero | Gate is safe but useless | Lower threshold or add fused features |
| Threshold too loose | Exit rate high but fidelity poor | Gate accepts boundary tokens | Raise threshold, add hard-token detector |
| Boundary blindness | Hard tokens exit early | Difficulty signal missing | Add entropy/moment/velocity features |
| Laminar under-skip | Easy tokens still run full depth | Threshold not adapted to clusters | Cluster-conditioned thresholds |
| Layer tap mismatch | Earlier/later layers behave nonsensically | Wrong hidden tap or norm | Verify layer indexing and apply final norm before lm_head |
| Feature overhead | Logical skip works but wall-clock slower | Feature extraction too expensive | Move features into C++ and measure tokens/joule |
| Distribution drift | Top-1 matches but KL/top-k drifts | Exact match too coarse | Track KL/top-k before deployment |
| Context drift | Gate works short-context only | Feature distribution shifts with length | Calibrate per context band |
