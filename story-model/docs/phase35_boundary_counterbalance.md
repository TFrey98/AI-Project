# Phase 35: compositional local-state boundary counterbalance

Phase 34k localized the paired `or`/`ro` boundary effect to
`proposal_key(h_current_token)`. Across both source-token strata and both the
Phase 34d and Phase 34i checkpoints, the local-key intervention accounted for
essentially the complete contextual effect while the pooled resolver query
accounted for none. The next controlled variable is therefore training
composition, not another representation feature or decoder rule.

Phase 35 keeps the successful Phase 33c resolver frozen and restores the
pre-geometry Phase 34d proposer and loss. It adds a tokenizer-aware route
curriculum in which the same local BPE identity receives balanced `B:route`
and `I:route` labels inside matched prompt pairs.

## Controlled boundary

- Restart from the eligible Phase 33c `best.pt`, never a Phase 34 checkpoint.
- Preserve the legacy joint typed-BIO head and Phase 34d boundary-transition
  objective at weight 1.
- Add no token-ID, byte-ID, token-width, token-end, factorized-head, query, or
  decoder feature.
- Preserve the 1,000-update pilot ceiling, optimizer, learning-rate schedule,
  batch size 2, gradient accumulation 4, and eight sampled rows per update.
- Exactly one of four microbatches is Phase 35 data. The other three retain
  one Phase 31 and one Phase 32 row each. Every update therefore contains
  three Phase 31, three Phase 32, and two Phase 35 rows.
- Rotate the Phase 35 microbatch through the four accumulation positions so
  update order does not become a hidden variable.
- Keep `or` and `ro` out of both new focus-token pools. They remain an inherited
  Phase 31 regression target, not a hand-coded training fix.
- Reserve eight viable two-byte BPE identities for training/validation and
  four different identities for lexical/transfer evaluation.
- Exclude any focus identity whose bytes occur in the fixed route forms or
  split markers. This prevents one identity from appearing inside another
  identity's generated candidate value.
- Require every focus identity to receive equal B and I support. A held-out
  focus identity may not occur anywhere in a Phase 35 training route value.
- Recompute counts over each completed split and require global B/I counts to
  exactly match the counts from rows assigned to that identity. Background O
  occurrences are recorded in the manifest but are not required to balance.
- Select checkpoints only when both the retained Phase 31/32 validation panel
  and the Phase 35 validation panel pass the original exact-span floors.

Phase 33c remains the last accepted resolver checkpoint throughout this phase.

## Build and freeze the counterbalance

Counterbalance schema version 2 supersedes the pre-training version 1 build,
which could count only rows assigned to a focus token and miss occurrences in
other generated routes. Version 1 manifests are invalid and must be rebuilt;
no checkpoint may cite them.

The builder reads the actual Phase 33c tokenizer rather than assuming which
strings form atomic BPE tokens. It also requires the frozen Phase 34k summary,
recomputes its premise, checks the tokenizer hash, and verifies the exact
Phase 31 lexical and transfer hashes used by the intervention.

From `story-model/`:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/build_boundary_counterbalance_dataset.py \
  --phase33-checkpoint checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt \
  --phase34k-summary runs/phase34k-context-path-intervention.summary.json
```

Do not train unless the builder reports:

- eight training and four held-out focus identities with zero overlap;
- `or` and `ro` excluded;
- all five source splits built from complete Phase 31 scene-route pairs;
- equal B/I support for every focus identity; and
- zero incidental positive-label collisions after whole-split recomputation;
- `Phase 35 boundary counterbalance dataset: passed`.

The generated manifest freezes every output hash, source hash, tokenizer hash,
focus-token ID and byte string, label count, and the Phase 34k premise hash.
It also freezes the filler vocabulary, assigned counts, global counts, and the
rule that global positive counts must equal assigned positive counts.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The tests cover Phase 34k premise rejection, atomic focus-token selection,
matched-pair transformation, equal per-token B/I support, held-out value-token
disjointness, the fixed 3/3/2 update composition, retained-plus-counterbalance
checkpoint selection, bidirectional focus confusion, provenance failures, and
every Phase 35 decision branch.

## Wiring smoke

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_boundary_counterbalance_smoke.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt
```

The smoke is a wiring test. Confirm:

- MPS is selected;
- startup reports boundary counterbalance version 2 and no new architecture;
- the focus pools are 8/4 with zero overlap;
- each logged update samples two of eight rows from the counterbalance;
- retained and counterbalance validation metrics are printed separately;
- losses and gradients are finite; and
- `final.pt` plus either `best.pt` or `diagnostic-ineligible.pt` is written.

Do not use a smoke checkpoint for the pilot.

## Bounded pilot

Restart from Phase 33c:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_boundary_counterbalance_pilot.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt
```

Stop at 1,000 updates. Use `best.pt` if the joint retained/new-data validation
gate creates one. Otherwise use `diagnostic-ineligible.pt` only for the frozen
diagnostics below; it cannot be promoted.

## Frozen retained and refreshed-token audits

First rerun the unchanged Phase 31/32 tag audit on the Phase 35 checkpoint:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_explicit_offset_tag_confusion.py \
  --checkpoint checkpoints/explicit_offset_boundary_counterbalance_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --batch-size 8 \
  --progress-every 25 \
  --output runs/phase35-retained-tag-audit.summary.json
```

Then evaluate all five Phase 35 splits, including each held-out focus identity
separately:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_boundary_counterbalance.py \
  --checkpoint checkpoints/explicit_offset_boundary_counterbalance_pilot/best.pt \
  --data-dir data/character/boundary_counterbalance \
  --device mps \
  --batch-size 8 \
  --output runs/phase35-boundary-counterbalance-audit.summary.json
```

If no eligible `best.pt` exists, substitute the pilot's
`diagnostic-ineligible.pt` in both commands.

Recompute the decision independently against the frozen Phase 34d audit:

```bash
.venv/bin/python scripts/gate_boundary_counterbalance.py \
  --baseline runs/phase34d-boundary-audit.summary.json \
  --candidate runs/phase35-retained-tag-audit.summary.json \
  --counterbalance runs/phase35-boundary-counterbalance-audit.summary.json \
  --output runs/phase35-boundary-counterbalance-decision.json
```

## Decision rules

The comparison requires matching checkpoint hashes, steps, tokenizers, and
counterbalance manifests. Every held-out focus token must have at least 20 B
and 20 I observations separately in both lexical and transfer.

For the refreshed lexical and transfer splits:

- per-token B-to-I and I-to-B must each be at most 2%;
- exact span precision and recall must each be at least 98%;
- answer-candidate recall must be at least 99.5%;
- positive-byte type accuracy must be at least 99%;
- offset validity must be 100%; and
- proposal overflow must be zero.

For every retained Phase 31/32 aggregate and per-skill cell:

- B-to-I may not increase by more than two percentage points from Phase 34d;
- O-to-I may not increase by more than half a percentage point;
- positive-byte type accuracy may not fall by more than one point;
- end spill must remain at or below 1%; and
- Phase 31 lexical and transfer `scene_route` B-to-I must be at most 2%.

| Branch | Meaning |
|---|---|
| `boundary_counterbalance_diagnostic_pass` | Refreshed identities compose and retained cells remain safe; run the unchanged full Phase 34 gate |
| `boundary_counterbalance_generalization_insufficient` | The data change does not transfer to new local token identities; inspect per-token errors without selecting another architecture |
| `boundary_counterbalance_rejected` | The original target remains broken or retained cells regress; reject the training mixture |
| `boundary_counterbalance_span_gate_ineligible` | Diagnostics are interpretable but no jointly eligible validation checkpoint exists; do not extend training |
| `invalid_boundary_counterbalance_comparison` | Repair checkpoint, manifest, focus support, or audit provenance |

Every branch keeps `checkpoint_promotion_authorized: false`.

## Inherited full Phase 34 gate

Only `boundary_counterbalance_diagnostic_pass` authorizes the existing
predicted-candidate evaluation:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/evaluate_explicit_offset_candidate_proposer.py \
  --checkpoint checkpoints/explicit_offset_boundary_counterbalance_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --batch-size 8 \
  --progress-every 25 \
  --output runs/phase35-explicit-offset-proposer.jsonl

.venv/bin/python scripts/gate_explicit_offset_candidate_proposer.py \
  --summary runs/phase35-explicit-offset-proposer.summary.json \
  --phase33-summary runs/phase33c-count-conditioned.summary.json
```

Promotion is authorized only if that complete inherited gate passes. A
successful diagnostic alone does not replace Phase 33c.
