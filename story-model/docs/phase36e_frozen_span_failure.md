# Phase 36e: frozen span-failure audit

Phase 36d reported "identity transfer passes, complete spans fail." That
aggregate is not an explanation. Phase 36e changes no training, weights,
decoder, or thresholds — it compares Phase 35, Phase 36b, and Phase 36d on
the identical Phase 36a crossed-identity panel and asks *what specifically*
deteriorated.

Registered because "recovery relative to Phase 35 is not recovery to
acceptable absolute quality": Phase 35's own transfer baseline averaged
10.3% exact-span precision and 27.1% recall, so a small positive delta
against it leaves substantial span failure unexplained.

## What it measures

- **Row-level transitions** against the Phase 35 baseline: previously
  correct spans that become wrong, previously wrong spans that recover, and
  rows wrong under both. Counts, not just precision/recall.
- **Per-gold-span error taxonomy**: `missing_entirely`, `misplaced_start`,
  `early_ending` (I→O), `fragmentation` (I→B), `late_ending`, `correct` —
  located relative to the row's focus token.
- **Spurious predicted spans**: predicted spans sharing zero bytes with any
  gold span. Note what this does and does not mean — a predicted span that
  overlaps the gold span at all is excluded from this bucket regardless of
  type or boundary error, so "spurious" means positioned elsewhere
  entirely, not "near miss."
- **Collapsed positive-vs-outside discrimination**, independent of B/I
  identity. The conditional B/I consistency objective never directly
  constrains the O logit, so correct B-vs-I discrimination can coexist with
  the model deciding the wrong bytes belong to a span at all.
- **Checkpoint-step reporting**, explicit and un-suppressed: the audited
  checkpoints were selected at different steps and any comparison stays
  qualified by that.

## Code

| File | Role |
|---|---|
| `src/story_model/span_failure_analysis.py` | Pure classification functions — no model or decoder code |
| `tests/test_span_failure_analysis.py` | 17 unit tests covering every taxonomy branch and both confusion helpers |
| `scripts/audit_frozen_span_failure.py` | Orchestration: loads three checkpoints, builds the panel, aggregates |
| `scripts/census_spurious_span_fragments.py` | Non-cherry-picked census of every spurious span, by byte length and prompt section |

## Run

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/audit_frozen_span_failure.py \
  --phase35-checkpoint checkpoints/explicit_offset_boundary_counterbalance_pilot/diagnostic-ineligible.pt \
  --phase36b-checkpoint checkpoints/explicit_offset_paired_identity_invariance_pilot/diagnostic-ineligible.pt \
  --phase36d-checkpoint checkpoints/explicit_offset_clean_anchor_identity_invariance_pilot/diagnostic-ineligible.pt \
  --device mps --batch-size 8 \
  --output runs/phase36e-frozen-span-failure.summary.json

PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/census_spurious_span_fragments.py \
  --checkpoint checkpoints/explicit_offset_clean_anchor_identity_invariance_pilot/diagnostic-ineligible.pt \
  --label phase36d --device mps --batch-size 8 \
  --output runs/phase36e-spurious-census-phase36d.json
```

## Findings

Panel: 14 identities × 2 splits × 200 rows = 5,600 rows. Only 159 of those
rows were ever correct under Phase 35 to begin with.

| Transition vs. Phase 35 | Phase 36b | Phase 36d |
|---|---|---|
| still_correct | 46 | 89 |
| regressed | 113 | 70 |
| newly_fixed | 6 | 1 |
| still_wrong | 5,435 | 5,440 |

Phase 36d roughly doubles the survival rate of previously-correct rows
(56% vs. 29%). The absolute correct count stays tiny either way.

**The targeted span is often geometrically close.** Per-gold-span: 59–65%
exactly correct on lexical, 26–28% on transfer, with `early_ending`
dominant on transfer (60–65%) and `fragmentation` roughly halved under
Phase 36d (13% → 6–7.5%).

**Low precision is driven by spurious spans, not by the target span.**
~3.5 spurious spans per row (Phase 36d), ~4.4 (Phase 36b). Byte lengths are
overwhelmingly tiny: 1–2 bytes account for 83.7% (Phase 36d) and 77.6%
(Phase 36b) of all spurious spans.

**Location, normalized by section size, not raw count.** Actual per-row
section lengths are `character` 279B, `relationship` 128B, `scene` ~175B,
`world_facts` ~51B, `memories` ~123B, `conversation` ~73B. Spurious spans
concentrate in `world_facts` and `conversation` (87–91% combined) — the two
*smallest* content sections — while `character`, the largest, holds
approximately none (70 events in Phase 36b, zero in Phase 36d). Section
length does not explain the pattern; if anything it runs opposite.

**Locally, discrimination is excellent.** In a window around the focus
token itself, O-vs-positive error is 0.02%–0.5% across both checkpoints and
all identity groups. The background errors are happening away from the
mechanism under test.

## Limitations recorded, not papered over

- Every row in this panel uses one fixed route-form scaffold
  (`CROSSED_ROUTE_VARIANT=0`) and all gold spans fall in a single 17–24
  byte length bucket. Neither scaffold nor span length is a variable in this
  data, so neither is reported as a breakdown axis.
- Swap exposure is reported at identity-group level only; no per-row swap
  log was recorded during training, so group membership is a proxy, not
  literal per-row incidence.
- A one-byte predicted span does not imply a one-byte input token — the
  byte-level decoder can extract part of a longer token. These histograms
  alone cannot establish swap-pool exposure. Phase 36f addresses this.
