# Phase 36g: bounded clean-anchor consistency-weight ablation

Phase 36d established that clean-anchor supervision preserves the tested
identity repair while reducing the damage seen under fully supervised swaps.
Complete-span quality remained insufficient. The remaining straightforward
variable is the strength of the consistency objective itself.

The immediate question is precise: **can we keep the identity repair with
less pressure from the consistency objective?**

## Arms

| Arm | Consistency weight | Checkpoint |
|---|---|---|
| Clean-anchor reference | 1.0 | `checkpoints/explicit_offset_clean_anchor_identity_invariance_pilot/diagnostic-ineligible.pt` (step 800) |
| Reduced-strength candidate | 0.1 | `checkpoints/explicit_offset_consistency_weight_0p1_pilot/step-800.pt` |

The 0.1 value is a deliberate tenfold reduction, **not a claimed optimum**.

Both arms restart from Phase 33c and retain begin-and-inside pairing,
original-only (clean-anchor) supervision, the same swap pool, training
mixture, architecture, decoder, optimizer, learning-rate schedule, and seed
(1337). `tests/test_phase36g_consistency_weight.py` asserts the two configs
are byte-identical apart from the weight, the checkpoint directory, and
`comparison_step`.

### Reference-arm reuse

The existing Phase 36d pilot is reused as the λ=1.0 arm because its
provenance satisfies the comparison: step 800, seed 1337, λ=1.0, clean
anchor, begin-and-inside, Phase 33c parent, identical schedule and swap
pool. The gate re-verifies every one of these fields against the candidate
and rejects the comparison if any differs.

## Predeclared comparison step

Phase 36g introduces `train.comparison_step`, a new config key. When set,
training saves `step-<N>.pt` from the state evaluated at exactly that step,
independent of whether checkpoint selection would have kept it. This exists
so two arms are compared at one fixed step rather than at whatever step each
arm's selection logic happened to retain.

Validation (`_load_config`): must be a positive integer, must not exceed
`max_steps`, and must land on an evaluation step — otherwise the save would
never fire and the comparison would silently not exist.

Phase 36g uses `comparison_step: 800`, inside the unchanged 1,000-update
ceiling.

## Run

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_consistency_weight_0p1_smoke.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt

PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_consistency_weight_0p1_pilot.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt

PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_explicit_offset_tag_confusion.py \
  --checkpoint checkpoints/explicit_offset_consistency_weight_0p1_pilot/step-800.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps --batch-size 8 --progress-every 25 \
  --output runs/phase36g-retained-tag-audit.summary.json

PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_consistency_weight_ablation.py \
  --phase35-checkpoint checkpoints/explicit_offset_boundary_counterbalance_pilot/diagnostic-ineligible.pt \
  --reference-checkpoint checkpoints/explicit_offset_clean_anchor_identity_invariance_pilot/diagnostic-ineligible.pt \
  --candidate-checkpoint checkpoints/explicit_offset_consistency_weight_0p1_pilot/step-800.pt \
  --device mps --batch-size 8 \
  --output runs/phase36g-consistency-weight-ablation.summary.json

.venv/bin/python scripts/gate_consistency_weight_ablation.py \
  --crossed-summary runs/phase36g-consistency-weight-ablation.summary.json \
  --baseline-retained runs/phase35-retained-tag-audit.summary.json \
  --candidate-retained runs/phase36g-retained-tag-audit.summary.json \
  --output runs/phase36g-consistency-weight-ablation.decision.json
```

## Outcomes judged separately

| Outcome | Branch | Interpretation |
|---|---|---|
| Identity transfer holds; span quality recovers | `reduced_consistency_promising` | Lower consistency strength is a promising improvement |
| Span quality recovers; identity transfer fails | `reduced_consistency_tradeoff` | The tested weight exposes a tradeoff; it does **not** prove every intermediate weight would fail |
| Identity transfer holds; spans remain damaged | `reduced_consistency_no_span_recovery` | This reduction provides no span-quality recovery |
| Both worsen | `reduced_consistency_rejected` | Reject the candidate |
| Provenance or arm mismatch | `invalid_consistency_weight_comparison` | Repair before interpreting |

"Span quality recovers" means reaching Phase 35 compatibility: no
identity/split cell drops more than 2 points in exact-span precision or
recall against the Phase 35 baseline. Recovering to Phase 35's span quality
establishes **compatibility with that baseline** — promotion still requires
the absolute quality floors and the inherited Phase 34 gate.

## Reporting contract

Unlike the Phase 36c/36d gates, **checkpoint eligibility does not rename the
scientific outcome**. The branch reports what the metrics did. Retained
metrics (`retained_ok`), transfer (`transfer_ok`), span quality (`span_ok`),
and eligibility (`candidate_checkpoint_eligible`) are reported as four
independent fields. `full_phase34_evaluation_authorized` requires the
promising branch **and** retained safety **and** an eligible checkpoint.
`checkpoint_promotion_authorized` is `false` under every branch.

An absent eligible checkpoint remains substantive, but it is not itself a
retained-metric regression.

## Result

**Branch: `reduced_consistency_rejected`.** Both outcomes worsened.

Arm equivalence verified by the gate: `invalid_reasons: []` — the two arms
differ only in the consistency weight and were compared at the same
predeclared step 800.

**Identity transfer failed.** Three of the 14 registered identities broke
under λ=0.1 — and they are exactly the previously-severe identities that
λ=1.0 had repaired:

| Identity | Split | λ=1.0 B→I | λ=0.1 B→I |
|---|---|---|---|
| 493 (`ho`, held-out) | lexical | 0.000 | **0.831** |
| 493 (`ho`, held-out) | transfer | 0.000 | **0.435** |
| 357 (`ro`, legacy) | lexical | 0.000 | **0.583** |
| 357 (`ro`, legacy) | transfer | 0.003 | **0.335** |
| 433 (`ld`, held-out) | lexical | 0.000 | **0.086** |
| 433 (`ld`, held-out) | transfer | 0.000 | **0.075** |

The other 11 identities held. The repair degrades exactly where it was
hardest to obtain.

**Span quality did not recover.** 35 identity/split cells still regress more
than 2 points against Phase 35. The tenfold reduction bought nothing on
complete spans.

**Retained metrics regressed.** `scene_route` B→I rose to 3.12% (lexical)
and 4.39% (transfer), now failing the 2% ceiling that λ=1.0 passed at 0.27%
on both splits.

**Reported independently:** `transfer_ok: False`, `span_ok: False`,
`retained_ok: False`, `candidate_checkpoint_eligible: False`. No eligible
`best.pt` was produced, consistent with every pilot in this chain; that is
recorded separately and is not counted as a metric regression.

### Interpretation

The consistency objective's strength is load-bearing for the identity
repair: weakening it tenfold undoes the repair on the hardest identities
while providing no span-quality recovery whatsoever.

Because this landed on `rejected` rather than `tradeoff`, the finding is
more informative than a tradeoff would have been. A tradeoff would have
motivated searching intermediate weights for a better operating point. Here
span quality did not move at all, so **consistency strength is not the
operative variable for complete-span quality**. Searching between 1.0 and
0.1 could still locate the minimum weight sufficient for the identity
repair, but that is a different question and should not be expected to
address the span blocker.

The standing caveat applies: this tests one value. It does not prove every
intermediate weight would fail the identity repair.

### Status

Phase 33c remains the accepted checkpoint. The λ=0.1 candidate is rejected;
the λ=1.0 clean-anchor design from Phase 36d remains the better-supported
pairing configuration. Complete-span extraction remains the unresolved
project blocker, and is now known not to be reachable by tuning consistency
strength.

## Deferred

Persistent background errors are **not** addressed here. They are deferred
to a separately registered training experiment, to be defined after Phase
36g so its effects remain distinguishable from consistency strength. See
`docs/phase36f_background_error_attribution.md`.
