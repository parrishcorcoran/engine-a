# Result Schema

Use this schema for every Engine A host run. The goal is to make results
comparable across models, prompt classes, layers, and gates.

## Required Metadata

```yaml
run_id:
date:
host:
model:
model_path:
dtype:
device:
prompt_file:
prompt_class:
context_tokens:
engine_a_commit:
command:
```

## Required Metrics

| Metric | Meaning | Good Direction |
|---|---|---|
| `gate` | Gate mode: confidence, margin, stability, fused | Compare |
| `exit_layer` | Layer where the logical exit is tested | Earlier saves more |
| `threshold` | Gate threshold | Sweep |
| `tokens` | Number of evaluated token positions | Higher confidence |
| `exit_rate` | Fraction of tokens accepted for early exit | Higher if fidelity holds |
| `fidelity` | Accepted exits matching final top-1 token | Higher |
| `false_exit_rate` | Wrong accepted exits over all tokens | Lower |
| `top1_agreement` | Mid-layer top-1 equals final top-1 | Higher |
| `topk_agreement` | Mid-layer top-1 appears in final top-k | Higher |
| `accepted_topk_agreement` | Accepted exits that stay within final top-k | Higher |
| `topk_overlap` | Jaccard overlap between mid and final top-k sets | Higher |
| `avg_layers_skipped` | Expected layers skipped per token | Higher if fidelity holds |
| `skip_gain` | Logical layer-skip fraction | Higher if fidelity holds |
| `avg_score` | Gate score average | Diagnostic |
| `avg_entropy` | Mid-layer entropy average | Lower for laminar prompts |

## Diagnosis Labels

Use exactly one primary label:

- `parser/eval failure`
- `baseline instability`
- `no calibration`
- `false plateau`
- `threshold too strict`
- `threshold too loose`
- `boundary blindness`
- `laminar under-skip`
- `layer tap mismatch`
- `layer signal absent`
- `late-only signal`
- `distribution drift`
- `context drift`
- `feature overhead`
- `shallow signal`
- `first-order signal`
- `partial unsafe signal`
- `weak signal`

## Result Log Template

```markdown
# Engine A Run

## Metadata

- Run ID:
- Date:
- Host:
- Model:
- Commit:
- Prompt class:
- Context tokens:
- Command:

## Best Row

| gate | layer | threshold | exit_rate | fidelity | false_exit_rate | avg_layers_skipped | topk_overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| | | | | | | | |

## Calibration

Paste calibration buckets here.

## Diagnosis

Primary label:

Why:

## Next Single Change

Only one change:
```

## Comparison Rule

When comparing two gates, do not compare raw exit rate alone. Compare at matched
fidelity bands:

- 90 percent fidelity.
- 95 percent fidelity.
- 97 percent fidelity.
- 99 percent fidelity.

The better gate is the one with higher `avg_layers_skipped` and lower
`false_exit_rate` inside the same fidelity band.
