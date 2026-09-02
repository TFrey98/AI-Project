# Phase 34d: explicit boundary-transition supervision

Phase 34c isolated the Phase 34 failure to boundary generalization rather than
global outside-tag pressure. Gold B was predicted as I on 17.6% of Phase 31
lexical starts and 19.7% of Phase 31 transfer starts, while O-to-I remained
small. The same audit also found frequent I spill onto the first outside byte
after a gold span.

Phase 34d changes one training variable: it adds explicit supervision at both
span transitions. It does not change the architecture, candidate annotations,
global O/B/I class weights, permissive decoder, resolver, or datasets.

## Objective

The original weighted BIO loss remains intact:

```text
O = 0.05, every typed B = 1.0, every typed I = 0.5
```

Two unweighted cross-entropy terms are added:

- `boundary_start_loss`: typed B prediction at every gold span's first byte.
- `boundary_end_loss`: O prediction at the first gold outside byte following
  each span.

The combined loss is:

\[
L = L_{BIO} + 1.0 \times \frac{L_{start} + L_{end}}{2}
\]

If a batch contains only one boundary category, the average contains only the
available term. If it contains no spans, the auxiliary loss is zero. The
objective adds no parameters.

## Controls

- Restart from the eligible Phase 33c checkpoint, never a Phase 34 or smoke
  checkpoint.
- Preserve seed 1337 and the Phase 34 data sampling.
- Preserve permissive BIO decoding.
- Preserve all class weights and model parameters.
- Do not rebuild the Phase 31 or Phase 32 datasets.
- Do not add plays or other corpus material during this experiment.
- Do not promote a diagnostic checkpoint.

## Tests

From `story-model/`:

```bash
.venv/bin/python -m pytest -q
```

The Phase 34d tests verify that the auxiliary gradient touches only gold start
and immediate end positions, composes correctly with BIO loss, adds no model
parameters, preserves Python 3.9-compatible configuration, and enforces the
post-pilot decision branches.

## Wiring smoke

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_boundary_supervision_smoke.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt
```

The smoke is a wiring test, not a convergence gate. It passes when:

- startup reports `typed_byte_BIO+boundary_transition` and weight `1`;
- the trainable parameter list is unchanged from Phase 34;
- logged start and end position counts are both nonzero;
- total, BIO, start, and end losses are finite;
- gradients are finite; and
- `final.pt` plus either `best.pt` or `diagnostic-ineligible.pt` is written.

The smoke does not need to reach the original 98% exact-span eligibility
floors in 100 updates. Do not use its checkpoint for the pilot.

## Bounded pilot

If the smoke wiring passes, restart from Phase 33c:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/train_explicit_offset_candidate_proposer.py \
  --config configs/explicit_offset_boundary_supervision_pilot.yaml \
  --warm-start checkpoints/count_conditioned_unified_typed_span_resolver_pilot/best.pt
```

The pilot remains capped at 1,000 updates. Do not continue beyond that ceiling.
Use `best.pt` if the run creates an eligible checkpoint. Otherwise use
`diagnostic-ineligible.pt` only for the following audit; it remains
non-promotable.

## Boundary audit and decision

Run the unchanged Phase 34c audit on the selected Phase 34d checkpoint:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_explicit_offset_tag_confusion.py \
  --checkpoint checkpoints/explicit_offset_boundary_supervision_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --batch-size 8 \
  --output runs/phase34d-boundary-audit.summary.json
```

If there is no `best.pt`, substitute
`explicit_offset_boundary_supervision_pilot/diagnostic-ineligible.pt`. Then
compare against the existing Phase 34c baseline:

```bash
.venv/bin/python scripts/gate_boundary_supervision.py \
  --baseline runs/phase34c-tag-confusion-audit.summary.json \
  --candidate runs/phase34d-boundary-audit.summary.json \
  --output runs/phase34d-boundary-decision.json
```

The diagnostic gate requires:

- B-to-I at or below 2% in every aggregate and per-skill Phase 31 lexical and
  transfer cell;
- at least 50% relative end-spill reduction in every aggregate and per-skill
  dataset/split cell;
- no cell's B-to-I rate increasing by more than two percentage points;
- positive-byte type accuracy losing at most one percentage point per cell;
- Phase 34d checkpoint metadata declaring objective version 1 and weight 1;
  and
- the training panel marking the candidate checkpoint eligible.

Possible branches are:

| Branch | Meaning |
|---|---|
| `boundary_diagnostic_pass` | Boundary targets and the original training-panel eligibility pass; run the full Phase 34 evaluation |
| `boundary_tags_fixed_span_gate_ineligible` | Boundary diagnostics pass but exact-span eligibility does not; stop |
| `start_fixed_end_insufficient` | B-to-I clears but end spill does not; stop |
| `end_fixed_start_insufficient` | End spill clears but Phase 31 lexical/transfer B-to-I does not; stop |
| `boundary_supervision_insufficient` | Neither boundary target clears; stop |
| `boundary_supervision_rejected` | Positive type accuracy regresses; reject the objective |
| `invalid_boundary_supervision_comparison` | Checkpoint provenance or audit cells do not match |

Only `boundary_diagnostic_pass` authorizes the expensive predicted-candidate
evaluation. It still does not promote the checkpoint.

## Full Phase 34 gate

After `boundary_diagnostic_pass`, run:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/evaluate_explicit_offset_candidate_proposer.py \
  --checkpoint checkpoints/explicit_offset_boundary_supervision_pilot/best.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --batch-size 8 \
  --progress-every 25 \
  --output runs/phase34d-explicit-offset-proposer.jsonl
```

```bash
.venv/bin/python scripts/gate_explicit_offset_candidate_proposer.py \
  --summary runs/phase34d-explicit-offset-proposer.summary.json \
  --phase33-summary runs/phase33c-count-conditioned.summary.json
```

Checkpoint promotion is authorized only if this original Phase 34 gate passes
in full. Otherwise retain Phase 33c as the last accepted resolver checkpoint.
