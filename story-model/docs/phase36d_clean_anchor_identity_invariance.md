# Phase 36d: clean-anchor paired identity invariance

Phase 36c proved the begin-only ablation did not localize the damage from
Phase 36b: even with zero `I:route` pairs generated (1,557 B swaps, 0 I
swaps, verified against the checkpoint's own recorded counts), exact-span
precision fell in all 28 identity/split cells, recall fell in 26/28, and the
mean gold-`I` begin-minus-inside logit margin shifted 1.33 logits toward
`B` — despite no inside position ever being paired. The damage is therefore
not about *which position* gets paired.

Phase 36d changes the other candidate variable: whether the swapped copy
receives its own direct supervised loss at all. It restores Phase 36b's
begin-and-inside pairing and keeps λ = 1.0, but removes every supervised
tag/type/boundary contribution from the swapped forward pass. The swapped
copy still exists and still participates in the JSD consistency term; it
just never appears in `output.loss` itself.

```
L = L_original_full_supervision + λ · L_JSD
```

## Frozen contract

- Restart from the same eligible Phase 33c `best.pt` used by Phase 35/36b/36c.
- Restore begin-and-inside pairing (`begin_only=False`), matching Phase 36b's
  position eligibility, not Phase 36c's restriction.
- Keep λ = 1.0, the same broad frozen swap pool, sampling frequency,
  optimizer, schedule, batch size 2, accumulation 4, and 1,000-update
  ceiling.
- Register `paired_identity_invariance_version: 3` and
  `identity_invariance_supervision_mode: clean_anchor`.
- The mechanism: after `paired_identity_invariance_batch` builds the
  augmented batch, `mask_swapped_supervision` sets every position of every
  swapped row's target to `IGNORE_TAG` before the forward pass. Because the
  model's tag/boundary losses use `F.cross_entropy(..., ignore_index=...)`,
  a fully-masked row contributes to neither the loss sum nor its normalizing
  count — `output.loss` becomes mathematically identical to computing it on
  the original batch alone.

## Correctness test (non-negotiable)

`tests/test_phase36d_clean_anchor_identity_invariance.py` proves, on a real
(tiny) model, that with `identity_invariance_loss_weight = 0` the clean-anchor
path's loss and every parameter's gradient are numerically identical
(`atol=1e-6`) to the unmodified Phase 35 base trainer run on the same
original batch. This guards against a masking or doubled-batch-averaging bug
silently rescaling the supervised objective. Do not train Phase 36d without
this test passing.

## Smoke and pilot

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_clean_anchor_identity_invariance_smoke.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt

PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_clean_anchor_identity_invariance_pilot.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt
```

Confirm the smoke run reports `identity-invariance supervision mode:
clean_anchor` at startup and that `swapped` diagnostic loss logs near zero
throughout (it should — the swapped copy is unsupervised, so its logged
"swapped supervised loss" diagnostic reflects an untrained/unused signal, not
a trained one). Do not evaluate a smoke checkpoint. Stop the pilot at 1,000
updates.

## Frozen evaluation

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_explicit_offset_tag_confusion.py \
  --checkpoint checkpoints/explicit_offset_clean_anchor_identity_invariance_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps --batch-size 8 --progress-every 25 \
  --output runs/phase36d-retained-tag-audit.summary.json

PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_clean_anchor_identity_invariance.py \
  --phase35-checkpoint checkpoints/explicit_offset_boundary_counterbalance_pilot/diagnostic-ineligible.pt \
  --phase36d-checkpoint checkpoints/explicit_offset_clean_anchor_identity_invariance_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase35-data-dir data/character/boundary_counterbalance \
  --device mps --batch-size 8 \
  --output runs/phase36d-clean-anchor-identity-invariance.summary.json

.venv/bin/python scripts/gate_clean_anchor_identity_invariance.py \
  --crossed-summary runs/phase36d-clean-anchor-identity-invariance.summary.json \
  --baseline-retained runs/phase35-retained-tag-audit.summary.json \
  --candidate-retained runs/phase36d-retained-tag-audit.summary.json \
  --output runs/phase36d-clean-anchor-identity-invariance.decision.json
```

If no eligible `best.pt` exists, the frozen diagnostics may substitute
`diagnostic-ineligible.pt`, but the gate's pass branch requires
`checkpoint_eligible: true`.

## Gate comparisons

- **Identity transfer**: all 14 registered identities (8 trained, 4
  held-out, 2 legacy — broadened from Phase 36c's 12-identity check after
  legacy `ro` was the one miss a 12-identity check would have caught) must
  be ≤2% B-to-I in both lexical and transfer.
- **Complete-span quality**: exact-span precision and recall compared
  against Phase 35, not Phase 36b or 36c.
- **Retained Phase 31/32 behavior**: compared against the Phase 35 baseline,
  same tolerances as every prior phase in this chain.
- **Checkpoint eligibility**: requires an actual eligible `best.pt`.

## Decision branches

| Branch | Meaning |
|---|---|
| `clean_anchor_identity_invariance_diagnostic_pass` | Transfer, span quality vs. Phase 35, and retained safety all hold with an eligible checkpoint → authorizes only the inherited full Phase 34 gate |
| `clean_anchor_identity_invariance_transfer_failed` | At least one of the 14 registered identities exceeds 2% B-to-I in either split |
| `clean_anchor_identity_invariance_span_still_damaged` | Transfer holds but span quality is still worse than Phase 35 → the damage is not explained by double supervision alone; inspect the JSD term itself next |
| `clean_anchor_identity_invariance_regressed_retained` | Retained Phase 31/32 cells regress beyond tolerance, or no eligible checkpoint exists |
| `invalid_clean_anchor_identity_invariance_comparison` | Provenance, panel, or support mismatch |

Every branch keeps `checkpoint_promotion_authorized: false`; only the
explicit pass branch sets `full_phase34_evaluation_authorized: true`.
