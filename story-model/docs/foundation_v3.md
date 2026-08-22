# Phase 24: controlled foundation-v3 capacity experiment

## Decision

Foundation-v2 lowered exact held-out performance on the home corpus from
`1.9352` to `1.7032` bits/byte.  Phase 24 keeps that exact corpus, validation
split, BPE vocabulary, random seed, and 4,096 sampled tokens per update while
testing whether a larger model and a 512-token window improve prose semantics.

| Property | Foundation-v2 | Foundation-v3 |
|---|---:|---:|
| Parameters | 2,863,616 | 11,424,672 |
| Layers | 6 | 8 |
| Embedding width | 192 | 384 |
| Attention heads | 6 | 6 |
| GELU-equivalent feed-forward width | 768 | 1,024 |
| Context | 256 | 512 |
| Batch size | 16 | 8 |
| Tokens/update | 4,096 | 4,096 |

The larger model starts with random weights.  Its layer shapes differ from
foundation-v2, so `--warm-start` is intentionally invalid.  It instead uses
`--tokenizer-from` to recover the exact 512-token BPE mapping from
`checkpoints/transformer_foundation_v2/best.pt` without loading that model's
weights or optimizer.  Omitting this flag would retrain BPE on the expanded
corpus and confound the capacity experiment with a tokenizer change.

Every new checkpoint records and prints SHA-256 identities for the tokenizer,
corpus manifest, training split, and validation split.  Compare those lines
across computers before comparing metrics.

## 1. Unit tests

Run from the `story-model/` directory:

```bash
.venv/bin/python -m pytest -q
```

## 2. MPS smoke test

```bash
.venv/bin/python scripts/run_with_monitor.py \
  --name foundation-v3-smoke \
  -- \
  .venv/bin/python -u -m story_model.train \
  --config configs/transformer_foundation_v3_smoke.yaml \
  --tokenizer-from checkpoints/transformer_foundation_v2/best.pt
```

The smoke gate passes when all of the following are true:

- device is `mps`;
- parameters are `11,424,672` and vocabulary is `512`;
- tokenizer/corpus SHA-256 lines are present;
- loss and gradient norms remain finite and loss declines;
- all 200 updates finish without memory pressure or a monitor hitch.

Do not use the smoke checkpoint as the new foundation.

## 3. Two-thousand-update pilot

Run this only after the smoke gate passes:

```bash
.venv/bin/python scripts/run_with_monitor.py \
  --name foundation-v3-pilot \
  -- \
  .venv/bin/python -u -m story_model.train \
  --config configs/transformer_foundation_v3_pilot.yaml \
  --tokenizer-from checkpoints/transformer_foundation_v2/best.pt
```

Proceed to the full run when validation loss has a clear downward trend,
training remains numerically stable, MPS memory stays bounded, and sustained
throughput is acceptable.  The pilot is a wiring/performance decision gate;
it is not expected to beat the fully trained foundation-v2 after only 2,000
updates.

## 4. Full run

```bash
.venv/bin/python scripts/run_with_monitor.py \
  --name foundation-v3-train \
  -- \
  .venv/bin/python -u -m story_model.train \
  --config configs/transformer_foundation_v3.yaml \
  --tokenizer-from checkpoints/transformer_foundation_v2/best.pt
```

The full budget is 30,000 updates with validation every 1,000 updates and
early stopping after six consecutive non-improvements.  Use
`checkpoints/transformer_foundation_v3/best.pt`, not `final.pt`, for selection.

## 5. Exact comparison

```bash
.venv/bin/python scripts/run_with_monitor.py \
  --name foundation-v3-val \
  -- \
  .venv/bin/python -u -m story_model.evaluate \
  --checkpoint checkpoints/transformer_foundation_v3/best.pt \
  --data-config configs/transformer_foundation_v3.yaml \
  --split val \
  --device mps
```

The quantitative acceptance target is exact validation below the current
`1.7032` bits/byte baseline.  A lower number is necessary but not sufficient:
fixed-prompt samples must also show better meaning across adjacent sentences,
not merely cleaner spelling and syntax.  Only after both gates pass should the
generic dialogue bridge be rebuilt from foundation-v3 and Vera-specific
authoring resume.
