# Phase 34i: factorized boundary and type heads

Phase 34h found that every width-2/offset-0 failure in the target
`scene_route` cells came from the BPE tokens `or` and `ro`. Both tokens had a
strong `I:route` training prior and neither had appeared at a familiar Phase
31 span start. That evidence is descriptive but not contrastive: the focus
bucket contains no unaffected token identity. The formal Phase 34h decision
therefore selected a factorized boundary/type head rather than authorizing a
token-specific data intervention.

Phase 34i tests one architectural variable. The old joint labels
`B:<type>`/`I:<type>` are replaced during training by:

- one shared three-class boundary head: `O`, `B`, or `I`; and
- one type head trained only on positive (`B` or `I`) bytes.

This removes the direct competition between `B:route` and `I:route`. Exact
token identities remain diagnostic outputs, never loss targets or gate
inputs.

## Objective

The frozen resolver and byte-state construction are unchanged. Given one
shared byte state, the proposer predicts independent boundary and type
logits:

\[
L = L_{O/B/I} + L_{type\mid positive}
  + 1.0 \times \frac{L_{start} + L_{end}}{2}
\]

`L_{O/B/I}` retains the Phase 34 weights `O=0.05`, `B=1.0`, and `I=0.5`.
`L_type` is unweighted and is computed only where the gold boundary is B or
I. The Phase 34d start/end terms now operate on the shared boundary logits.

For the unchanged decoder, typed BIO logits are composed so that:

- the maximum typed-B logit exactly equals the shared B logit;
- the maximum typed-I logit exactly equals the shared I logit; and
- the winning positive type is the independent type-head argmax.

Therefore type identity cannot turn an independently predicted B into I.

## Fixed controls

- Restart from the same eligible Phase 33c checkpoint, not any Phase 34
  checkpoint.
- Preserve seed 1337, tokenizer, datasets, sampling, byte-state inputs,
  permissive decoder, optimizer, schedule, and 1,000-update ceiling.
- Preserve the Phase 34d boundary-transition objective and weight 1.
- Do not add token-width, token-end, token-ID, or byte-ID features.
- Do not add or rebalance training examples in this experiment.
- Do not promote a diagnostic-ineligible checkpoint.

The legacy joint tag head remains in the state dictionary, initialized exactly
as in Phase 34d, but is frozen. This preserves the control RNG trajectory and
keeps old checkpoints/loaders compatible. Only the factorized boundary and
type heads replace it as trainable output parameters.

## Tests and primitive

From `story-model/`:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/overfit_explicit_offset_candidate_proposer.py \
  --factorized-boundary-type
```

The tests verify target decomposition, independent argmax preservation,
separate gradient flow, zero safe type loss on all-outside batches, exact RNG
and shared-parameter preservation, checkpoint versioning, controlled config
diffs, and every decision branch. The factorized primitive uses a 6,000-step
maximum (the legacy primitive remains at 3,000) and must reach exact
train-span F1, answer recall, and offset validity. This diagnostic ceiling
does not alter either real-data configuration.

## Wiring smoke

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_factorized_head_smoke.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt
```

The smoke is a wiring test, not a convergence gate. It passes when:

- startup reports factorized boundary/type version 1;
- the trainable list contains `proposal_boundary` and `proposal_type`, while
  the legacy `proposal_tag` is absent;
- boundary-class, positive-byte type, start, and end losses are finite;
- positive type positions and both boundary position counts are nonzero;
- gradients are finite; and
- `final.pt` plus either `best.pt` or `diagnostic-ineligible.pt` is written.

Do not use the smoke checkpoint for the pilot.

## Bounded pilot

Restart from the same Phase 33c checkpoint:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_factorized_head_pilot.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt
```

Stop at the 1,000-update ceiling. Use `best.pt` if it exists; otherwise use
`diagnostic-ineligible.pt` only for the following diagnostic audit.

## Frozen audit and decision

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_explicit_offset_tag_confusion.py \
  --checkpoint checkpoints/explicit_offset_factorized_head_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --batch-size 8 \
  --progress-every 25 \
  --output runs/phase34i-factorized-head-audit.summary.json
```

If no eligible `best.pt` exists, substitute the pilot's
`diagnostic-ineligible.pt`. Then compare it with the frozen Phase 34d audit:

```bash
.venv/bin/python scripts/gate_factorized_candidate_proposer.py \
  --baseline runs/phase34d-boundary-audit.summary.json \
  --candidate runs/phase34i-factorized-head-audit.summary.json \
  --output runs/phase34i-factorized-head-decision.json
```

The diagnostic passes only when:

- every aggregate and per-skill Phase 31 lexical/transfer B-to-I rate is at
  most 2%;
- the target width-2 B-to-I rate is at most 2%;
- end spill remains at most 1% in every aggregate and per-skill cell;
- no cell's B-to-I rate increases by more than two percentage points;
- no cell's O-to-I rate increases by more than half a percentage point;
- no cell loses more than one percentage point of positive-byte type
  accuracy;
- the checkpoint declares boundary objective version 1, weight 1, and
  factorized-head version 1, with neither geometry feature; and
- the training panel produced an eligible checkpoint.

The audit also reports tokens 270 and 357 if present, but those identities do
not affect the decision.

| Branch | Meaning |
|---|---|
| `factorized_head_diagnostic_pass` | Boundary, safety, and checkpoint gates pass; run the unchanged full Phase 34 gate |
| `factorized_head_span_fixed_but_ineligible` | Boundary targets pass but the training panel remains ineligible; stop |
| `factorized_head_insufficient` | Shared boundaries do not fix held-out route starts; stop and design a training-only compositional counterbalance with refreshed held-out values |
| `factorized_head_rejected` | A retained boundary or type cell regressed; reject the architecture |
| `invalid_factorized_head_comparison` | Checkpoint provenance, audit cells, or the target bucket is invalid |

Only `factorized_head_diagnostic_pass` authorizes the expensive end-to-end
evaluation. It does not authorize checkpoint promotion.

## Full unchanged gate

After the diagnostic pass:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/evaluate_explicit_offset_candidate_proposer.py \
  --checkpoint checkpoints/explicit_offset_factorized_head_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --batch-size 8 \
  --progress-every 25 \
  --output runs/phase34i-explicit-offset-proposer.jsonl
```

```bash
.venv/bin/python scripts/gate_explicit_offset_candidate_proposer.py \
  --summary runs/phase34i-explicit-offset-proposer.summary.json \
  --phase33-summary runs/phase33c-count-conditioned.summary.json
```

Promotion is authorized only if the complete inherited Phase 34 gate passes.
Otherwise Phase 33c remains the last accepted resolver checkpoint.
