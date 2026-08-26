# Phase 28: semantic and factorized transfer

## Why Phase 28 exists

Phase 27 fixed the repetition collapse and taught the model to use evidence on
unseen combinations of familiar words. At 120 sampled rows, train and
compositional validation had similar grounding scores, while combined
lexical-plus-paraphrase transfer remained near zero. Exact response matching
also failed even when a generated sentence could select the right fact using
different wording.

Phase 28 measures those failures separately. It keeps the 11.4M-parameter
foundation-v3 Transformer, response-only objective, adjacent counterfactual
pairs, and pair-aware batching. It does not start Vera-specific training and
does not authorize a 10,000-update run.

## Data design

The builder produces five splits:

| Split | Vocabulary | Question wording | Purpose |
| --- | --- | --- | --- |
| `train` | Expanded development | Expanded development | Paired training |
| `val` | Development | Development | Held-out combinations |
| `lexical` | Unseen | Familiar constructions | Isolate word transfer |
| `paraphrase` | Familiar | Held-out constructions | Isolate wording transfer |
| `transfer` | Unseen | Held-out constructions | Combined transfer |

Exact question strings remain held out on the paraphrase axes. The expanded
training templates expose semantic vocabulary such as *hue*, *passable*, and
*traversable* in different constructions so this small model is asked to
generalize compositionally rather than infer an entirely unseen task label.

`answer_keys.json` records the expected value and the counterfactual foil for
every context. Semantic scoring accepts a differently worded response when it
selects the expected color or route. Exact match and reference similarity are
still reported as surface-form diagnostics. Route answers may name both ways;
the scorer recognizes common forms that reject the blocked route and select
the usable one. Review nontrivial `semantic_status` cases qualitatively because
this is a controlled entity-selection scorer, not a general entailment model.

## 1. Build and audit

From `story-model/`:

```bash
.venv/bin/python scripts/build_semantic_transfer_dataset.py

.venv/bin/python scripts/audit_character_dataset.py \
  --checkpoint checkpoints/transformer_foundation_v3/best.pt \
  --data data/character/semantic_transfer/train.jsonl \
  --block-size 1024 \
  --min-examples 3200 \
  --min-conversations 1600 \
  --min-tag-examples 1600 \
  --required-tag canon \
  --required-tag relationship \
  --required-tag scene \
  --required-tag uncertainty \
  --required-tag world_fact \
  --max-dropped-turns 0

.venv/bin/python scripts/audit_character_dataset.py \
  --checkpoint checkpoints/transformer_foundation_v3/best.pt \
  --data data/character/semantic_transfer/val.jsonl \
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

Repeat the second audit for `lexical.jsonl`, `paraphrase.jsonl`, and
`transfer.jsonl`. The builder verifies adjacency, one-line evidence changes,
different pair targets, split isolation, and complete semantic answer keys.

## 2. Wiring smoke test

```bash
.venv/bin/python -m story_model.train \
  --config configs/transformer_semantic_transfer_smoke.yaml \
  --warm-start checkpoints/transformer_foundation_v3/best.pt
```

Require MPS, `paired_conversation`, finite losses and gradients, decreasing
loss, and both checkpoint files. This 100-update run is not a quality result.

## 3. Bounded pilot

```bash
.venv/bin/python -m story_model.train \
  --config configs/transformer_semantic_transfer_pilot.yaml \
  --warm-start checkpoints/transformer_foundation_v3/best.pt
```

The pilot has a 1,000-update ceiling and early stopping. Use `best.pt`, not
`final.pt`, for every following command. Evaluate full teacher-forced loss on
all five splits, for example:

```bash
for split in train val lexical paraphrase transfer; do
  .venv/bin/python scripts/evaluate_character.py \
    --checkpoint checkpoints/transformer_semantic_transfer_pilot/best.pt \
    --data data/character/semantic_transfer/$split.jsonl \
    --device mps
done
```

Run complete counterfactual pairs through greedy generation and evidence
ablation. Thirty pairs per skill gives 120 rows per split:

```bash
.venv/bin/python scripts/diagnose_neutral_instruction.py \
  --checkpoint checkpoints/transformer_semantic_transfer_pilot/best.pt \
  --train-data data/character/semantic_transfer/train.jsonl \
  --val-data data/character/semantic_transfer/val.jsonl \
  --extra-data lexical=data/character/semantic_transfer/lexical.jsonl \
  --extra-data paraphrase=data/character/semantic_transfer/paraphrase.jsonl \
  --extra-data transfer=data/character/semantic_transfer/transfer.jsonl \
  --device mps \
  --pairs-per-skill 30 \
  --max-new-tokens 80 \
  --output runs/phase28-semantic-transfer-diagnosis.jsonl

.venv/bin/python scripts/evaluate_counterfactual_semantics.py \
  --diagnostics runs/phase28-semantic-transfer-diagnosis.jsonl \
  --answer-keys data/character/semantic_transfer/answer_keys.json \
  --output runs/phase28-semantic-transfer-scored.jsonl

.venv/bin/python scripts/gate_semantic_transfer.py \
  --generation-summary \
    runs/phase28-semantic-transfer-diagnosis.summary.json \
  --semantic-summary \
    runs/phase28-semantic-transfer-scored.summary.json
```

## Acceptance criteria

Exact response rate is reported but is not a gate. The gate asks whether the
correct value was selected, both sides of each pair were correct, generation
ended cleanly without loops, and evidence improved the answer.

| Split | Semantic rows | Semantic pairs | End | Loops | Context helped | Advantage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | >=90% | >=80% | >=95% | <=5% | >=75% | >=0.12 |
| Compositional val | >=75% | >=60% | >=95% | <=5% | >=75% | >=0.12 |
| Lexical only | >=60% | >=40% | >=95% | <=5% | >=60% | >=0.08 |
| Paraphrase only | >=60% | >=40% | >=95% | <=5% | >=60% | >=0.08 |
| Combined transfer | >=40% | >=25% | >=95% | <=5% | >=50% | >=0.05 |

The pattern determines the next change:

- lexical-only failure calls for more value diversity and token exposure;
- paraphrase-only failure calls for more intent-preserving constructions;
- both isolated axes passing but combined transfer failing calls for mixed
  lexical/paraphrase counterfactuals during training;
- compositional validation failure despite good train semantics is evidence
  to test an objective or capacity change;
- all axes passing justifies expanding the paired design to the other neutral
  conversation skills before any Vera voice training.

Do not run 10,000 updates based only on lower teacher-forced validation loss.
