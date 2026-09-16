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

## Investigation record

Phase 36d's checkpoint is rejected — `checkpoint_promotion_authorized:
false` under every branch, and it landed on
`clean_anchor_identity_invariance_regressed_retained` (no eligible
`best.pt`). That gate outcome stands unchanged. But the follow-on frozen
diagnostics (Phase 36e's row-level/error-taxonomy audit and Phase 36f's
background-error attribution audit, both no-training comparisons against
Phase 35 and Phase 36b on the identical crossed-identity panel) established
a real, separable experimental contribution inside that rejected
checkpoint, worth recording on its own terms.

**Defensible finding.** At the matched 800-step checkpoints, clean-anchor
training preserved the 14-identity B→I pass, roughly halved fragmentation,
and reduced background false spans by 20.8% relative to Phase 36b.
Background errors remained near the Phase 35 baseline, and complete-span
quality remained insufficient for promotion.

**Three qualifications:**

1. **A pre-existing error does not imply an architecture defect.** Phase
   36f established that background false positives were already present in
   Phase 35 — it did not establish their origin in Phase 34d, and it does
   not show that fixing them requires architectural changes. Training
   composition, supervision, and loss weighting can change this behavior
   with the architecture and decoder held fixed. Addressing it would
   require a separately registered experiment; the frozen architecture does
   not prohibit that work.
2. **"Returns approximately to baseline" is supported; "matches or beats
   every section" is not.** Two exceptions: Phase 36d `conversation` errors
   exceed Phase 35 (1.39% vs. 1.21% lexical, 2.80% vs. 2.71% transfer), and
   Phase 36b `scene` errors are slightly *lower* than Phase 35 on lexical
   (0.56% vs. 0.58%). The overall −2.1% span-count difference should not be
   treated as an established improvement, given the checkpoint-step
   mismatch (Phase 35 is step 1000; Phase 36b/36d are both step 800) and
   the lack of replication.
3. **The fragmentation improvement is established against Phase 36b, not
   Phase 35.** That makes it a meaningful ablation result — removing
   swapped-copy supervision improved this outcome relative to full
   supervision. It does not demonstrate that clean-anchor training improves
   fragmentation over the original (pre-Phase-36b) counterbalance baseline.
   The lexical-split precision/recall regression documented in Phase 36e
   still matters independently and is unresolved.

| Question | Supported conclusion |
|---|---|
| Can identity invariance repair the tested B/I generalization gap? | Yes, on the registered panel. |
| Does clean-anchor supervision improve on fully supervised swaps? | Yes, in these matched-checkpoint results. |
| Did identity invariance originate the background-error problem? | No; it was already present in Phase 35. |
| Have the underlying causes of background errors and fragmentation been fully identified? | No; their behavior is better separated, but their mechanisms remain partly unresolved. |
| Is Phase 36d eligible for promotion? | No. |

**Disposition:** documenting a successful component inside a rejected
experiment is appropriate here. Clean-anchor is the better-supported
pairing design for any future paired-identity-invariance experiment in this
chain. The original gate decision is preserved unchanged — no promotion.
Complete-span extraction remains the unresolved project blocker.
