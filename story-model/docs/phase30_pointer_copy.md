# Phase 30: byte-level evidence pointer

Phase 29 showed that adding hundreds of sparse values did not teach the model
to reproduce a novel value from evidence. Phase 30 restores the Phase 28
curriculum and changes one variable: an explicit pointer/copy path.

Each record adds an optional `copy_value`. The prompt keeps its normal BPE
encoding, while target occurrences of that value are losslessly represented by
the tokenizer's guaranteed raw UTF-8 byte tokens. The pointer attends to
virtual byte positions inside merged prompt tokens. Therefore a value remains
copyable even when BPE represents `"uses lime t"` in the prompt and `"lime."`
in the response with unrelated token IDs.

```text
P(final) = p(generate) * P(vocabulary)
         + (1 - p(generate)) * P(copy prompt byte)
```

Generated response tokens never become copy sources. This prevents the copy
path from amplifying its own repetition.

## Build and verify

From `story-model/`:

```bash
pytest -q
python scripts/overfit_pointer_copy.py
python scripts/build_pointer_copy_dataset.py
```

Expected rows: 3,200 train and 400 each for val, lexical, paraphrase, and
transfer. Audit every split with the foundation-v3 checkpoint and add
`--require-copy-supervision`. Every row must receive at least one copy target.

The regression suite contains an adversarial tokenizer with incompatible BPE
boundaries around `lime`. It must still produce four byte-copy targets.

## Smoke

```bash
python -m story_model.train \
  --config configs/transformer_pointer_copy_smoke.yaml \
  --warm-start checkpoints/transformer_foundation_v3/best.pt
```

Require MPS, finite losses and gradients, nonzero copy-token counts, declining
train/validation loss, and both checkpoints. The log must list
`copy_query.weight`, `copy_key.weight`, `copy_offset_embedding.weight`,
`copy_gate.weight`, and `copy_gate.bias` as new architecture parameters.

## Bounded pilot

Only after the smoke test passes:

```bash
python -m story_model.train \
  --config configs/transformer_pointer_copy_pilot.yaml \
  --warm-start checkpoints/transformer_foundation_v3/best.pt
```

The pilot is capped at 1,000 updates with early stopping. Evaluate `best.pt`
on train, val, lexical, paraphrase, and transfer using the existing Phase 28
semantic-transfer diagnostics and unchanged gates. Do not authorize a
10,000-update run unless all five splits pass.
