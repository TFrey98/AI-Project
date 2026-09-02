# Phase 34c: byte-tag confusion and loss-mass audit

Phase 34b rejected strict BIO decoding on the unchanged update-800 pilot
logits. Exact-span precision improved from 0.177 to 0.293, but the gain was
below the pre-registered 0.20 floor and Phase 31 lexical/transfer recall fell
by 0.059 and 0.070. The checkpoint must remain diagnostic-only.

Phase 34c is a no-training audit. It determines whether the next controlled
variable should be the outside-tag weight or explicit boundary-start
supervision. It does not change the proposer, decoder, class weights,
checkpoint, or Phase 34 gate.

## Fixed controls

- Read `diagnostic-ineligible.pt`, which represents pilot update 800.
- Run one proposer forward pass per batch.
- Keep permissive Phase 34 decoding as the runtime baseline.
- Keep the existing type-specific class weights collapsed as
  `O=0.05`, `B=1.0`, and `I=0.5`.
- Evaluate every eligible Phase 31 and Phase 32 row in all five splits.
- Continue excluding only the pre-existing `wrong_type` proposer cases.
- Do not promote the checkpoint or run the end-to-end Phase 34 gate.

## Run

From `story-model/`:

```bash
.venv/bin/python scripts/audit_explicit_offset_tag_confusion.py \
  --checkpoint checkpoints/explicit_offset_candidate_proposer_pilot/diagnostic-ineligible.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --batch-size 8 \
  --output runs/phase34c-tag-confusion-audit.summary.json
```

If MPS memory permits, `--batch-size 16` may reduce dispatch overhead. Batch
size changes throughput only; it does not change logits or decision rules.

## Measurements

The summary reports overall, per-dataset/split, and per-skill values for:

- gold and predicted byte counts after collapsing typed tags to O/B/I;
- the 3-by-3 O/B/I confusion matrix and row-normalized rates;
- gold B predicted as I;
- orphan I tags partitioned by the gold O/B/I class at the same byte;
- type-mismatched I tags partitioned by gold class;
- I-tag bleed on the gold O byte immediately before a span;
- I-tag spill on the first gold O byte after a span;
- positive-byte type accuracy;
- target-weight mass under the current O/B/I weights;
- realized weighted negative-log-likelihood mass by gold class;
- a count-balanced and an NLL-balanced diagnostic candidate for `w_O`.

The count-balanced outside weight is:

\[
w_O = \frac{N_B w_B + N_I w_I}{N_O}
\]

The NLL-balanced value replaces each count with that class's summed
unweighted negative log likelihood. Both values are diagnostics, not automatic
hyperparameter choices.

## Pre-registered decision

| Branch | Required observation | Next single variable |
|---|---|---|
| `boundary_start_objective_indicated` | Any dataset/split/skill cell has B-to-I rate at least 0.10 | Add only explicit boundary-start supervision; preserve weights and permissive decoding |
| `outside_weight_ablation_indicated` | At least 80% of orphan I tags fall on gold O and every cell has B-to-I at most 0.05 | Change only `w_O`; preserve B/I weights and permissive decoding |
| `mixed_or_ambiguous_boundary_errors` | Neither condition isolates one mechanism | Stop and inspect the per-cell report |
| `no_orphan_inside_problem_detected` | The audit observes no orphan I tags | Stop because the Phase 34b premise was not reproduced |

The boundary-start branch takes precedence when both excess I-on-O and B-to-I
errors are present. Increasing O pressure while true B bytes are already being
classified as I could further reduce recall, so an O-only run is authorized
only inside its narrow safety region.

Phase 34c never authorizes a decoder change, checkpoint promotion, or a
training change by itself. Its result selects the one variable to specify in a
separate bounded follow-up experiment.
