# Phase 36c: begin-only identity invariance

Phase 36b validated the core hypothesis: embedding-swap consistency drove all
four previously-severe identities (366, 433, 493, legacy `ro`) from severe
crossed B-to-I bias to essentially zero, in both lexical and transfer, while
all eight trained identities and the original `scene_route` B-to-I target
stayed fixed. But it broadly damaged complete-span precision/recall across
nearly every identity and introduced a small `supplied_fact` regression, and
never produced an eligible `best.pt`. Phase 36c isolates whether pairing at
`I:route` positions specifically — not the begin-position fix itself — was
the source of that span damage.

## Frozen contract

Every Phase 36b setting is unchanged except one: the swap-position filter.

- Same λ = 1.0, same broad frozen swap pool (498 tokens, 14 registered
  identities excluded, width-stratified), same pair sampling frequency,
  optimizer, LR schedule, 1,000-update ceiling, architecture, decoder, and
  Phase 33c restart checkpoint.
- Same combined-forward-pass mechanic: original and swapped rows are
  concatenated into one batch, one forward pass,
  `total_loss = output.loss + λ · JSD(original, swapped)`. Phase 36c does not
  decompose this into two forward passes or otherwise change what gradient
  the model receives.
- Register `paired_identity_invariance_version: 2` for this position-restricted
  variant. The swap pool's own schema version stays pinned at 1 — the pool
  itself is provably unchanged from Phase 36b, only which positions may draw
  from it changed.

The only functional change: `paired_identity_invariance_batch` is called with
`begin_only=True`, which restricts the candidate-position filter to
`gold_tag == begin_tag("route")`. `I:route` positions are never selected.
This is enforced twice — a hard assertion inside the training function itself,
and an independent post-hoc audit against the trained checkpoint's recorded
position counts.

Two diagnostic-only values are additionally logged per step, decomposed from
the single already-computed forward pass (no second forward pass, no
gradient effect): the supervised typed-BIO loss restricted to original rows,
and restricted to swapped rows.

## Smoke and pilot

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_begin_only_identity_invariance_smoke.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt

PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_begin_only_identity_invariance_pilot.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt
```

Do not evaluate a smoke checkpoint. Stop the pilot at 1,000 updates.

## Frozen evaluation

Run the standard retained tag-confusion audit on the Phase 36c checkpoint,
then the three-way crossed-identity audit against Phase 35 and Phase 36b:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_explicit_offset_tag_confusion.py \
  --checkpoint checkpoints/explicit_offset_begin_only_identity_invariance_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps --batch-size 8 --progress-every 25 \
  --output runs/phase36c-retained-tag-audit.summary.json

PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_begin_only_identity_invariance.py \
  --phase35-checkpoint checkpoints/explicit_offset_boundary_counterbalance_pilot/diagnostic-ineligible.pt \
  --phase36b-checkpoint checkpoints/explicit_offset_paired_identity_invariance_pilot/diagnostic-ineligible.pt \
  --phase36c-checkpoint checkpoints/explicit_offset_begin_only_identity_invariance_pilot/best.pt \
  --phase36a-summary runs/phase36a-crossed-boundary-identity.summary.json \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --phase35-data-dir data/character/boundary_counterbalance \
  --device mps --batch-size 8 \
  --output runs/phase36c-begin-only-identity-invariance.summary.json

.venv/bin/python scripts/gate_begin_only_identity_invariance.py \
  --crossed-summary runs/phase36c-begin-only-identity-invariance.summary.json \
  --baseline-retained runs/phase35-retained-tag-audit.summary.json \
  --candidate-retained runs/phase36c-retained-tag-audit.summary.json \
  --output runs/phase36c-begin-only-identity-invariance.decision.json
```

If no eligible `best.pt` exists, the frozen diagnostics may substitute
`diagnostic-ineligible.pt`, but the gate's pass branch requires
`checkpoint_eligible: true` — a diagnostic substitute cannot pass.

## Gate comparisons

- **Identity transfer**: all four previously-severe identities (366, 433,
  493, 357) and all eight trained identities must be ≤2% B-to-I in both
  lexical and transfer — an absolute floor, not a delta against Phase 36b.
- **Complete-span quality**: exact-span precision and recall compared against
  **Phase 35**, not the rejected Phase 36b, since Phase 36b's span metrics
  are already known-bad and would set an uninformative bar.
- **Retained Phase 31/32 behavior**: compared against the existing frozen
  Phase 35 baseline, including the 2-point `supplied_fact` tolerance carried
  over unchanged.
- **Checkpoint eligibility**: requires an actual eligible `best.pt`.

## Hard audit

`audit_begin_only_identity_invariance.py` independently verifies, not merely
reports:

1. The Phase 36c checkpoint's recorded position counts have zero `I:route`
   pairs and at least one `B:route` pair.
2. Pairing opportunity did not shrink: for each of Phase 31 train, Phase 32
   train, and the Phase 35 counterbalance train split, it recomputes — from
   the frozen data and the frozen swap pool, using the same eligibility
   function the training loop itself calls — how many rows have a valid
   begin-and-inside-eligible candidate versus a begin-only-eligible
   candidate, and requires the begin-only count is not smaller.
3. The swap pool's hash is identical between Phase 36b and Phase 36c
   checkpoints (proving nothing about the pool itself changed).
4. Tokenizer, architecture, boundary objective, and counterbalance-manifest
   provenance match across all three checkpoints (Phase 35, 36b, 36c).

## Decision branches

| Branch | Meaning |
|---|---|
| `begin_only_identity_invariance_diagnostic_pass` | Transfer, span quality vs. Phase 35, and retained safety all hold, with an eligible checkpoint → authorizes only the inherited full Phase 34 gate |
| `begin_only_identity_invariance_span_recovered_transfer_regressed` | Identity transfer breaks again under begin-only pairing → inside-position pairing was necessary for transfer; the ablation disproves itself |
| `begin_only_identity_invariance_transfer_preserved_span_still_damaged` | Transfer holds but span quality is still worse than Phase 35 → inside-position pairing was not the (sole) damaging component; the next variable is masking the swapped copy's supervised loss to the selected position, not lowering λ |
| `begin_only_identity_invariance_regressed_retained` | Retained Phase 31/32 cells regress beyond tolerance, or no eligible checkpoint exists → reject |
| `invalid_begin_only_identity_invariance_comparison` | Provenance, swap-pool, hard-audit, or panel mismatch |

Every branch keeps `checkpoint_promotion_authorized: false`; only the
explicit pass branch sets `full_phase34_evaluation_authorized: true`.
