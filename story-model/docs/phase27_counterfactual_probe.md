# Phase 27: counterfactual grounding probe

## Decision from Phase 26

Phase 26 did not fail because of context truncation, decoding, insufficient
updates, or demonstrated model-capacity limits. Its update-700 checkpoint
reproduced training responses with 0.986 similarity and used training context,
but achieved 0% exact match and zero context advantage on held-out examples.
It memorized mappings between familiar templates and responses.

Phase 27 therefore keeps the 11.4M-parameter foundation-v3 architecture and
tests a data-level remedy before expanding all ten instruction skills.
Neither Phase 26 checkpoint is a valid warm start.

## Experimental design

The probe contains only `scene_route` and `supplied_fact`. Every conversation
is an adjacent pair of examples. The serialized prompts in a pair are
identical except for one evidence line, and that line changes the required
answer. Evidence rotates across world facts, scene state, memory, and an older
conversation turn. Question and response templates are substantially more
varied than in the Phase 26 probe.

The three splits isolate different problems:

- `train`: paired counterfactual training examples;
- `val`: familiar vocabulary and templates in unseen combinations; and
- `transfer`: unseen vocabulary plus unseen question/response paraphrases.

The training sampler draws both members of a pair into every two-row
microbatch. With four-step gradient accumulation, each update sees four
complete counterfactual pairs.

## 1. Build and audit

From `story-model/`:

```bash
.venv/bin/python scripts/build_counterfactual_probe_dataset.py

.venv/bin/python scripts/audit_character_dataset.py \
  --checkpoint checkpoints/transformer_foundation_v3/best.pt \
  --data data/character/counterfactual_probe/train.jsonl \
  --block-size 1024 \
  --min-examples 1600 \
  --min-conversations 800 \
  --min-tag-examples 800 \
  --required-tag canon \
  --required-tag relationship \
  --required-tag scene \
  --required-tag uncertainty \
  --required-tag world_fact \
  --max-dropped-turns 0

.venv/bin/python scripts/audit_character_dataset.py \
  --checkpoint checkpoints/transformer_foundation_v3/best.pt \
  --data data/character/counterfactual_probe/val.jsonl \
  --block-size 1024 \
  --min-examples 400 \
  --min-conversations 200 \
  --min-tag-examples 200 \
  --required-tag canon \
  --required-tag relationship \
  --required-tag scene \
  --required-tag uncertainty \
  --required-tag world_fact \
  --max-dropped-turns 0
```

Repeat the validation audit for `transfer.jsonl`. The builder itself proves
that every pair is adjacent, shares one conversation identifier, has the same
question, requires different responses, and differs in exactly one serialized
prompt line.

## 2. Smoke run

Warm-start directly from foundation-v3:

```bash
.venv/bin/python -m story_model.train \
  --config configs/transformer_counterfactual_probe_smoke.yaml \
  --warm-start checkpoints/transformer_foundation_v3/best.pt
```

The log must report `character batch sampling: paired_conversation`, MPS,
finite losses and gradients, declining train/compositional-validation loss,
and both checkpoint files. This smoke run tests wiring only.

## 3. Pilot run

```bash
.venv/bin/python -m story_model.train \
  --config configs/transformer_counterfactual_probe_pilot.yaml \
  --warm-start checkpoints/transformer_foundation_v3/best.pt
```

Evaluate teacher-forced response loss on all three axes:

```bash
.venv/bin/python scripts/evaluate_character.py \
  --checkpoint checkpoints/transformer_counterfactual_probe_pilot/best.pt \
  --data data/character/counterfactual_probe/train.jsonl \
  --device mps

.venv/bin/python scripts/evaluate_character.py \
  --checkpoint checkpoints/transformer_counterfactual_probe_pilot/best.pt \
  --data data/character/counterfactual_probe/val.jsonl \
  --device mps

.venv/bin/python scripts/evaluate_character.py \
  --checkpoint checkpoints/transformer_counterfactual_probe_pilot/best.pt \
  --data data/character/counterfactual_probe/transfer.jsonl \
  --device mps
```

Then run balanced, free-running generation with evidence ablation:

```bash
.venv/bin/python scripts/diagnose_neutral_instruction.py \
  --checkpoint checkpoints/transformer_counterfactual_probe_pilot/best.pt \
  --train-data data/character/counterfactual_probe/train.jsonl \
  --val-data data/character/counterfactual_probe/val.jsonl \
  --transfer-data data/character/counterfactual_probe/transfer.jsonl \
  --device mps \
  --pairs-per-skill 5 \
  --max-new-tokens 80 \
  --output runs/phase27-counterfactual-diagnosis.jsonl

.venv/bin/python scripts/gate_counterfactual_probe.py \
  --summary runs/phase27-counterfactual-diagnosis.summary.json
```

## Acceptance and next decision

The automatic gate requires:

| Split | Rows exact | Pairs exact | End | Loops | Context helped | Context advantage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | >=90% | >=80% | >=95% | <=5% | >=80% | >=0.15 |
| Compositional val | >=70% | >=50% | >=95% | <=5% | >=75% | >=0.12 |
| Transfer | >=50% | >=30% | >=95% | <=5% | >=60% | >=0.08 |

Passing this probe justifies rebuilding all ten neutral skills with the same
paired design. It does not yet authorize Vera-specific training. Failure on
compositional validation after the paired curriculum is the first controlled
evidence that an objective or capacity change may be required. More Phase 26
updates and a 10,000-update Phase 27 run remain blocked.
