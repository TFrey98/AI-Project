# Phase 34e: begin-logit calibration and BPE geometry ablation

Phase 34d reduced end spill to approximately zero but left the Phase 31
`scene_route` lexical and transfer B-to-I error essentially unchanged. More
training or a larger start-loss coefficient is not indicated: the same
objective already fits train, validation, paraphrase, and Phase 32.

Phase 34e is a no-training ablation. It distinguishes three hypotheses:

1. The correct typed-B logit is present but needs one globally calibrated
   offset.
2. Failures concentrate at a particular BPE token/byte geometry.
3. Joint typed-BIO classification entangles boundary and type decisions and
   should be factorized.

The Phase 34d checkpoint, parameters, class weights, and permissive decoder
remain unchanged. Phase 34e writes JSON diagnostics only.

## Calibration protocol

The script evaluates global offsets added to every typed-B logit:

```text
0.00, 0.25, 0.50, ... 4.00
```

Bias selection uses deterministic train and validation panels only, with eight
examples per skill/action/case/candidate-width cell. Lexical, paraphrase, and
transfer examples cannot affect the selected value.

For every bias, the calibration panel measures exact-span precision/recall,
gold I predicted as B, gold O predicted as B, and positive-byte type accuracy.
The safety rules must pass on the pooled panel and independently in every
dataset/split and dataset/split/skill cell. Relative to zero bias, a value is
safe only when:

- exact-span precision loss is at most 0.005;
- exact-span recall loss is at most 0.005;
- I-to-B increase is at most 0.005;
- O-to-B increase is at most 0.0005; and
- positive-byte type accuracy loss is at most 0.01.

The largest safe value is selected. This deliberately tests the strongest
train/validation-supported calibration without tuning on held-out vocabulary.

## Run

Phase 34d must already be applied. From `story-model/`, run against the
checkpoint selected by the Phase 34d pilot. An ineligible checkpoint is valid
for this diagnostic and remains non-promotable:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python \
  scripts/ablate_begin_logit_calibration.py \
  --checkpoint checkpoints/explicit_offset_boundary_supervision_pilot/diagnostic-ineligible.pt \
  --phase31-data-dir data/character/typed_span_resolver \
  --phase32-data-dir data/character/expanded_typed_span_resolver \
  --device mps \
  --batch-size 8 \
  --progress-every 25 \
  --output runs/phase34e-begin-calibration.jsonl
```

If Phase 34d produced an eligible `best.pt`, use that path instead. The script
rejects checkpoints that do not declare boundary-objective version 1 and
weight 1.0.

The calibration grid reuses each panel forward pass. The full evaluation then
runs each record through the proposer once and decodes both zero-bias and
calibrated policies from those identical logits. Increasing `--batch-size` may
reduce MPS dispatch overhead if memory permits.

Outputs:

- `runs/phase34e-begin-calibration.jsonl`: one row per gold span start,
  including margins and BPE geometry.
- `runs/phase34e-begin-calibration.summary.json`: calibration selection,
  baseline/calibrated metrics for every split and skill, geometry tables, and
  the formal decision.

## Geometry measurements

At each gold start, the audit records:

- correct-type B-minus-I logit margin before and after calibration;
- baseline and calibrated predicted boundary class;
- BPE token width and the start byte's offset inside that token;
- whether the span starts at a token boundary or inside a merged token;
- the token prefix preceding the gold span;
- preceding-character class;
- prompt section and occurrence ordinal; and
- span byte length and word count.

The decision uses only BPE alignment, byte offset, and token width as geometry
causes. Preceding characters, sections, and span lengths remain descriptive.
A BPE bucket is considered explanatory only if:

- both the bucket and its complement contain at least 10% of target starts and
  at least 20 starts;
- it captures at least 50% of Phase 31 lexical/transfer `scene_route` B-to-I
  errors; and
- its error rate is at least twice the complement's rate.

This prevents a property shared by every example from being mistaken for a
causal partition.

## Decision

The held-out calibration branch requires every aggregate and per-skill Phase
31 lexical/transfer B-to-I rate to reach at most 2%. Across every dataset,
split, and skill, calibration may lose at most one percentage point of exact
span precision, exact span recall, or positive-byte type accuracy, and end
spill must remain at or below 1%.

| Branch | Interpretation |
|---|---|
| `global_begin_bias_indicated` | A positive train/validation-selected bias fixes held-out starts without violating safety; test that bias as the sole runtime change |
| `bpe_geometry_representation_indicated` | Calibration fails and a pre-registered BPE geometry bucket explains the errors; expose cross-token byte-boundary features next |
| `factorized_boundary_type_head_indicated` | Calibration fails without a geometry concentration; replace joint typed-BIO output with separate shared boundary and type heads |
| `invalid_begin_calibration_ablation` | The checkpoint is not the required Phase 34d experiment |

No branch promotes the Phase 34d checkpoint or authorizes training. Phase 34e
selects the next single controlled variable only. Do not run the full Phase 34
gate from this diagnostic result.
