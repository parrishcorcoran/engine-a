# Engine A + Engine B Coupling

Engine A may not be complete without Engine B.

This sounds counterintuitive because the old feature work found the compute
dimensions and memory/trajectory dimensions were not strongly correlated. But
lack of correlation does not mean lack of usefulness. It may mean the two
engines are complementary sensors of the same boundary state.

## The Hypothesis

Engine A asks:

> Has the token distribution collapsed enough to skip more compute?

Engine B asks:

> Is the current memory support collapsed enough to trust the local trajectory?

A token can look sharp locally while memory support is still ambiguous. That is
the false-plateau case:

- Mid-layer softmax is confident.
- The final layer later changes the answer.
- The token was not truly laminar; it was locally overconfident.

An Engine B-style memory tension signal can act as a refusal sensor for Engine
A. It does not need to be highly correlated with Engine A confidence. In fact,
it is more useful if it catches different failures.

## Cheap Or Free Coupling Signals

If Engine B is already running:

- `support_mass`: how much memory support is needed to preserve behavior.
- `support_entropy`: whether selected memory is concentrated or diffuse.
- `d_support`: whether memory support is contracting or expanding.
- `distractor_pressure`: whether competing support clusters remain active.
- `target/distractor ambiguity`: whether multiple memory wedges still survive.

If Engine B is not running yet, use cheap proxies:

- Recent hidden-state recurrence.
- Query/key similarity spread.
- Attention-output norm or attention entropy if already exposed.
- Token reuse / local recurrence.
- Difference between final-state and trajectory support scores.

## Integration Rule

Do not let Engine B force an exit. Let it veto unsafe exits.

Engine A should remain the compute-sharpness gate. Engine B should say:

> The token looks sharp, but memory is still turbulent. Do not exit yet.

That is cheap because vetoes can be low-dimensional and conservative. They do
not need perfect prediction; they only need to catch false plateaus.

## Testable Prediction

At matched exit rate or matched skip gain:

- A-only confidence gates will have more false exits.
- A-only guarded gates will be safer but may under-skip.
- A+B veto gates should reduce false exits versus free sharpness gates.
- If Engine B is already computed for memory pruning, the veto is nearly free.

Run:

```bash
python measurements/free_signal_budget.py --seeds 100
```

Expected:

- Engine B tension is weakly correlated with Engine A confidence.
- Free sharpness beats confidence-only.
- Engine B veto reduces false exits and false plateaus versus free sharpness.
- Stability still helps, but it may not be free on every runtime.

## Why This Matters

If Engine A only sees local token sharpness, it can be fooled by false
plateaus. If Engine B sees unresolved memory support, it can prevent those
exits. The two signals do not need to correlate. They need to disagree in the
right places.
