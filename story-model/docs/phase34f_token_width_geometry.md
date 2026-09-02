# Phase 34f: explicit token-width geometry

Phase 34e found that `token_width == 2` captures 91.3% of the remaining
Phase 31 lexical/transfer `scene_route` B-to-I errors. Its error rate was
59.3%, versus 3.0% outside that bucket. No positive global B-logit bias was
safe on the train/validation calibration panels.

The existing proposer already supplies the byte value and the byte offset
inside its BPE token. It does not explicitly supply the token's total width.
Phase 34f adds that one missing input and changes nothing else.

## Architecture

For every prompt token, its UTF-8 byte length indexes a learned embedding.
That embedding is added to every byte state belonging to the token:

```text
byte state = tanh(
    contextual token state
    + byte embedding
    + byte-offset embedding
    + token-width embedding
)
```

The width table has `source_width + 1` rows. It is initialized to zero, so
Phase 34f has the same initial logits as Phase 34d. Its construction also
preserves the random-number-generator state. With seed 1337, every existing
proposer parameter and the sampled-batch trajectory therefore begin from the
Phase 34d control state; only the new width table can separate the runs.

## Fixed controls

- Restart from the same eligible Phase 33c checkpoint, not a Phase 34d or
  Phase 34f checkpoint.
- Preserve the Phase 34d BIO and boundary-transition objective with weight 1.
- Preserve O/B/I class weights, permissive decoding, seed, datasets, sampling,
  tokenizer, learning-rate schedule, and update ceilings.
- Do not add an end-of-token flag, remaining-width feature, new data, or a
  factorized tag head in this experiment.
- Do not promote a diagnostic checkpoint.

## Tests

From `story-model/`:

```bash
.venv/bin/python -m pytest -q
```

The regression tests verify that the width-aware model initially matches the
Phase 34d logits and RNG trajectory exactly, the new table receives gradients,
legacy checkpoints remain loadable, configuration differs only by the width
version and output directory, and all formal decision branches remain fixed.

## Wiring smoke

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_token_width_geometry_smoke.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt
```

The smoke is a wiring test rather than a convergence gate. It passes when:

- startup reports `token-width geometry: version 1`;
- the only additional trainable parameter is
  `proposal_token_width_embedding.weight`;
- start and end position counts are nonzero;
- total, BIO, start, and end losses and gradients remain finite; and
- `final.pt` plus either `best.pt` or `diagnostic-ineligible.pt` is written.

Do not use the smoke checkpoint for the pilot.

## Bounded pilot

Restart from the same Phase 33c checkpoint:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_token_width_geometry_pilot.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt
```

The pilot remains capped at 1,000 updates. Use `best.pt` if it exists;
otherwise use `diagnostic-ineligible.pt` only for the diagnostic audit.

## Frozen boundary audit

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_explicit_offset_tag_confusion.py \
  --checkpoint checkpoints/explicit_offset_token_width_geometry_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --batch-size 8 \
  --progress-every 25 \
  --output runs/phase34f-token-width-audit.summary.json
```

If no eligible `best.pt` exists, substitute the pilot's
`diagnostic-ineligible.pt`. The audit now records the target token-width,
byte-offset, and alignment buckets in addition to the unchanged Phase 34c
metrics.

Compare it with the existing Phase 34d audit:

```bash
.venv/bin/python scripts/gate_token_width_geometry.py \
  --baseline runs/phase34d-boundary-audit.summary.json \
  --candidate runs/phase34f-token-width-audit.summary.json \
  --output runs/phase34f-token-width-decision.json
```

The diagnostic passes only when:

- every aggregate and per-skill Phase 31 lexical/transfer B-to-I rate is at
  most 2%;
- the target `token_width == 2` B-to-I rate is at most 2%;
- end spill remains at most 1% in every dataset/split and per-skill cell;
- no cell's B-to-I rate increases by more than two percentage points;
- no cell's O-to-I rate increases by more than half a percentage point;
- no cell loses more than one percentage point of positive-byte type accuracy;
- checkpoint metadata declares boundary-objective version 1, weight 1, and
  token-width-geometry version 1; and
- the training panel produced an eligible checkpoint.

Possible branches:

| Branch | Meaning |
|---|---|
| `token_width_geometry_diagnostic_pass` | Width geometry and retained controls pass; run the unchanged full Phase 34 gate |
| `token_width_fixed_span_gate_ineligible` | Boundary targets pass but the training panel remains ineligible; stop |
| `token_width_geometry_insufficient` | Width does not fix the registered target; test end-of-token geometry next |
| `token_width_geometry_rejected` | A retained boundary or type cell regressed; reject the change |
| `invalid_token_width_geometry_comparison` | Checkpoint provenance, audit cells, or the width-2 bucket is invalid |

Only `token_width_geometry_diagnostic_pass` authorizes the full evaluation. It
does not itself authorize checkpoint promotion.

## Full unchanged gate

After the diagnostic pass, run:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/evaluate_explicit_offset_candidate_proposer.py \
  --checkpoint checkpoints/explicit_offset_token_width_geometry_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --batch-size 8 \
  --progress-every 25 \
  --output runs/phase34f-explicit-offset-proposer.jsonl
```

```bash
.venv/bin/python scripts/gate_explicit_offset_candidate_proposer.py \
  --summary runs/phase34f-explicit-offset-proposer.summary.json \
  --phase33-summary runs/phase33c-count-conditioned.summary.json
```

Promotion is authorized only if this complete inherited gate passes.
