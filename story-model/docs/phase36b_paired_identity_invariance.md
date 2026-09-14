# Phase 36b: paired identity-invariance objective

Phase 36a returned `trained_identity_memorization_dominant`: every trained
identity reached zero crossed B-to-I error, while token IDs 366, 433, 493, and
357 remained severe. Phase 36b keeps the Phase 34d architecture and decoder
fixed and introduces one training variable: a paired identity-invariance loss
with weight 1.

## Frozen contract

- Restart from the same eligible Phase 33c `best.pt` used by Phase 35.
- Keep the resolver, backbone, proposer shapes, parameter count, and decoder
  unchanged.
- Keep `typed_byte_BIO + boundary_transition` at weight 1.
- Reuse the Phase 31/32/35 data and rotating 3/3/2 base-row composition.
- Keep optimizer, schedule, batch size 2, accumulation 4, and 1,000 updates.
- Add no dataset file, focus identity, geometry feature, or decoder rule.
- Register `PAIRED_IDENTITY_INVARIANCE_VERSION = 1` and loss weight 1.0.

For each sampled row containing a route span, the training loop chooses a
gold `B:route` or `I:route` token position and substitutes a different token
from the same byte-width class. The original and paired rows share all byte
labels and are forwarded together. Both receive the supervised loss. Their
conditional B/I predictions at byte offset zero additionally receive a
Jensen-Shannon consistency loss.

The replacement pool includes all non-special tokenizer identities except the
eight trained, four held-out, and legacy `or`/`ro` identities. It is grouped by
exact byte width. The full pool and canonical hash are stored in checkpoint
`extra`; the audit independently rebuilds them. Nothing is written to `data/`.

## Smoke and pilot

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_paired_identity_invariance_smoke.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt

PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_paired_identity_invariance_pilot.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt
```

Do not evaluate a smoke checkpoint. Stop the pilot at 1,000 updates.

## Frozen evaluation

Run the standard retained audit on the Phase 36b checkpoint, then:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_paired_identity_invariance.py \
  --phase35-checkpoint checkpoints/explicit_offset_boundary_counterbalance_pilot/diagnostic-ineligible.pt \
  --phase36b-checkpoint checkpoints/explicit_offset_paired_identity_invariance_pilot/best.pt \
  --phase36a-summary runs/phase36a-crossed-boundary-identity.summary.json \
  --phase36a-decision runs/phase36a-crossed-boundary-identity.decision.json \
  --device mps --batch-size 8 \
  --output runs/phase36b-paired-identity-invariance.summary.json

.venv/bin/python scripts/gate_paired_identity_invariance.py \
  --crossed-summary runs/phase36b-paired-identity-invariance.summary.json \
  --baseline-retained runs/phase35-retained-tag-audit.summary.json \
  --candidate-retained runs/phase36b-retained-tag-audit.summary.json \
  --output runs/phase36b-paired-identity-invariance.decision.json
```

`identity_invariance_diagnostic_pass` requires all four previously-severe
identities and all eight trained identities to pass the 2% crossed ceiling in
both splits, with no retained or complete-span regression. It authorizes only
the inherited full Phase 34 gate. Every branch keeps checkpoint promotion
unauthorized.
