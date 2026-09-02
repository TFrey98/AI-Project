# Phase 34g: explicit end-of-token geometry

Phase 34f supplied total BPE-token width directly, but the registered
`token_width == 2` Phase 31 lexical/transfer `scene_route` B-to-I rate remained
58.8%, essentially unchanged from the 59.3% Phase 34e baseline. Safety passed,
so the width input was harmless but insufficient.

Phase 34g tests the conjunction implicated by that result: whether the gold
span starts on the final byte of its current BPE token. It replaces the failed
width embedding rather than stacking another feature on it.

## Mandatory frozen premise audit

Before any training, derive the registered end-of-token relationship from the
existing Phase 34e start rows:

```bash
.venv/bin/python scripts/audit_token_end_premise.py \
  --start-rows runs/phase34e-begin-calibration.jsonl \
  --output runs/phase34g-token-end-premise.json
```

The premise passes only when:

- the target contains at least 100 starts and 20 B-to-I errors;
- token-end and non-token-end comparison buckets each contain at least 20
  starts;
- token ends capture at least 80% of all target B-to-I errors;
- token ends capture at least 90% of width-2 target errors; and
- the token-end error rate is at least five times the non-token-end rate.

`token_end_geometry_indicated` authorizes the smoke and bounded pilot. Any
other branch prohibits training and directs the next audit toward byte/token
identity. Both Phase 34g configs reference this report, and the trainer refuses
to start unless its decision explicitly authorizes training.

## Architecture

For each real byte position:

```text
is_token_end = byte_offset + 1 == token_width
```

A two-row embedding represents false/true. Row zero is a fixed padding row;
only the true token-end vector receives gradients. Both rows initialize to
zero, and construction preserves the RNG state, so initial logits and every
pre-existing parameter match the Phase 34d control.

The byte state becomes:

```text
byte state = tanh(
    contextual token state
    + byte embedding
    + byte-offset embedding
    + token-end embedding
)
```

## Fixed controls

- Restart from the eligible Phase 33c checkpoint, not Phase 34d, 34f, or a
  smoke checkpoint.
- Preserve the Phase 34d boundary-transition objective and weight 1.
- Preserve class weights, permissive decoding, tokenizer, datasets, seed,
  sampling, learning-rate schedule, and update ceilings.
- Do not retain the Phase 34f width embedding.
- Do not add remaining-width, token identity, new data, or a factorized head.
- Do not promote a diagnostic checkpoint.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Tests cover the frozen premise decision, zero-logit/RNG equivalence, gradient
flow only through the true end-marker row, mutual exclusion with width
geometry, legacy checkpoint loading, controlled configs, and every post-pilot
decision branch.

## Wiring smoke

After `token_end_geometry_indicated`:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_token_end_geometry_smoke.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt
```

The smoke passes when startup reports token-end geometry version 1, the sole
additional parameter is `proposal_token_end_embedding.weight`, all boundary
counts are nonzero, losses and gradients are finite, and the new smoke
directory receives `final.pt` plus a best or diagnostic checkpoint. The smoke
does not need to converge. Do not use it for the pilot.

## Bounded pilot

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_token_end_geometry_pilot.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt
```

Stop at 1,000 updates. Use `best.pt` if eligible; otherwise use
`diagnostic-ineligible.pt` only for the diagnostic audit.

## Post-pilot audit and decision

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_explicit_offset_tag_confusion.py \
  --checkpoint checkpoints/explicit_offset_token_end_geometry_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --batch-size 8 \
  --progress-every 25 \
  --output runs/phase34g-token-end-audit.summary.json
```

Substitute `diagnostic-ineligible.pt` if there is no eligible best checkpoint.
Then compare against the existing Phase 34d audit:

```bash
.venv/bin/python scripts/gate_token_end_geometry.py \
  --baseline runs/phase34d-boundary-audit.summary.json \
  --candidate runs/phase34g-token-end-audit.summary.json \
  --output runs/phase34g-token-end-decision.json
```

The target gate requires B-to-I at most 2% in every aggregate and per-skill
Phase 31 lexical/transfer cell, in the width-2 target bucket, and in the
token-end target bucket. Safety retains Phase 34f's end-spill, O-to-I,
B-to-I-regression, and type-accuracy limits. The checkpoint must also be
eligible.

| Branch | Meaning |
|---|---|
| `token_end_geometry_diagnostic_pass` | Registered targets and controls pass; run the unchanged full Phase 34 gate |
| `token_end_fixed_span_gate_ineligible` | Boundary targets pass but the training panel remains ineligible; stop |
| `token_end_geometry_insufficient` | The direct marker fails; stop scalar geometry experiments and test factorized boundary/type heads |
| `token_end_geometry_rejected` | A retained boundary or type cell regresses; reject the feature |
| `invalid_token_end_geometry_comparison` | Provenance, audit cells, or registered geometry buckets are invalid |

Only `token_end_geometry_diagnostic_pass` authorizes the existing full Phase
34 evaluation. Checkpoint promotion still requires that complete gate and the
unchanged Phase 33c oracle regression gate to pass.
