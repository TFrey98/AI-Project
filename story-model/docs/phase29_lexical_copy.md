# Phase 29: sparse lexical-copy curriculum

## Decision from Phase 28

Phase 28 separated lexical and paraphrase transfer. Paraphrase passed every
gate, but lexical transfer produced the following semantic statuses across 120
rows:

- `expected_only`: 13;
- `alternative_only`: 0; and
- `value_missing`: 107.

The combined-transfer split similarly missed both candidate values in 112 of
120 rows. The model is not choosing the wrong counterfactual value. It is
usually producing neither value and falling back to its familiar response
vocabulary. This identifies value copying/emission as the controlled failure.

Phase 29 keeps the foundation-v3 architecture, response-only loss, semantic
scorer, pair-aware batches, and Phase 28 acceptance thresholds. It changes
only the training curriculum. Warm-start from foundation-v3 again so the
comparison measures the data change rather than continued specialization of a
Phase 28 checkpoint.

## Curriculum design

Training contains 1,600 pairs per skill:

- 1,200 sparse-copy pairs use a pool of 384 compositional colors or 384
  compositional routes;
- 400 anchor pairs retain the familiar Phase 28 development vocabulary; and
- each pair still differs in exactly one evidence line and reverses the
  required answer.

Each compositional value combines familiar token pieces into a value such as
`cinder saffron` or `willow causeway`. The exact value appears only a handful
of times. This makes a closed memorized answer list ineffective while keeping
the task feasible for the existing BPE tokenizer: attend to evidence and
reproduce the supplied value.

The `val`, `lexical`, `paraphrase`, and `transfer` records are byte-for-byte
the same Phase 28 records when built with the defaults and seed 1337. Their
answer keys are also preserved. This makes Phase 28 and Phase 29 generation
results directly comparable.

## 1. Build and audit

From `story-model/`:

```bash
.venv/bin/python scripts/build_lexical_copy_dataset.py

.venv/bin/python scripts/audit_character_dataset.py \
  --checkpoint checkpoints/transformer_foundation_v3/best.pt \
  --data data/character/lexical_copy/train.jsonl \
  --block-size 1024 \
  --min-examples 6400 \
  --min-conversations 3200 \
  --min-tag-examples 3200 \
  --required-tag canon \
  --required-tag relationship \
  --required-tag scene \
  --required-tag uncertainty \
  --required-tag world_fact \
  --max-dropped-turns 0

.venv/bin/python scripts/audit_character_dataset.py \
  --checkpoint checkpoints/transformer_foundation_v3/best.pt \
  --data data/character/lexical_copy/val.jsonl \
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

Repeat the validation audit for `lexical.jsonl`, `paraphrase.jsonl`, and
`transfer.jsonl`. The builder reports how many copy-pool values were exposed
and their minimum, mean, and maximum training frequency.

## 2. Wiring smoke test

```bash
.venv/bin/python -m story_model.train \
  --config configs/transformer_lexical_copy_smoke.yaml \
  --warm-start checkpoints/transformer_foundation_v3/best.pt
```

Require MPS, `paired_conversation`, finite loss and gradients, declining loss,
and both checkpoint files. The 100 updates test wiring only.

## 3. Bounded pilot

```bash
.venv/bin/python -m story_model.train \
  --config configs/transformer_lexical_copy_pilot.yaml \
  --warm-start checkpoints/transformer_foundation_v3/best.pt
```

The 2,000-update ceiling provides at most 2.5 effective passes over the 6,400
training rows. Early stopping remains active. Do not use `final.pt` merely
because it has more updates; use the checkpoint with the best validation loss.

Evaluate teacher-forced loss on every split:

```bash
for split in train val lexical paraphrase transfer; do
  .venv/bin/python scripts/evaluate_character.py \
    --checkpoint checkpoints/transformer_lexical_copy_pilot/best.pt \
    --data data/character/lexical_copy/$split.jsonl \
    --device mps
done
```

Run 120 free-running rows per split, then apply the unchanged semantic scorer
and gate:

```bash
.venv/bin/python scripts/diagnose_neutral_instruction.py \
  --checkpoint checkpoints/transformer_lexical_copy_pilot/best.pt \
  --train-data data/character/lexical_copy/train.jsonl \
  --val-data data/character/lexical_copy/val.jsonl \
  --extra-data lexical=data/character/lexical_copy/lexical.jsonl \
  --extra-data paraphrase=data/character/lexical_copy/paraphrase.jsonl \
  --extra-data transfer=data/character/lexical_copy/transfer.jsonl \
  --device mps \
  --pairs-per-skill 30 \
  --max-new-tokens 80 \
  --output runs/phase29-lexical-copy-diagnosis.jsonl

.venv/bin/python scripts/evaluate_counterfactual_semantics.py \
  --diagnostics runs/phase29-lexical-copy-diagnosis.jsonl \
  --answer-keys data/character/lexical_copy/answer_keys.json \
  --output runs/phase29-lexical-copy-scored.jsonl

.venv/bin/python scripts/gate_semantic_transfer.py \
  --generation-summary runs/phase29-lexical-copy-diagnosis.summary.json \
  --semantic-summary runs/phase29-lexical-copy-scored.summary.json
```

## Acceptance and comparison

Use the Phase 28 thresholds unchanged. The primary causal comparison is the
lexical split:

| Metric | Phase 28 | Phase 29 gate |
| --- | ---: | ---: |
| Semantic rows | 10.8% | >=60% |
| Semantic pairs | 3.3% | >=40% |
| Context helped | 39.2% | >=60% |
| Context advantage | +0.034 | >=0.08 |
| Value missing | 89.2% | Report; should fall sharply |

Paraphrase must remain above its existing gate so the copy curriculum does not
trade away wording generalization. Train and validation semantic accuracy
should rise together; pair accuracy should then rise approximately with the
square of row accuracy.

If unfamiliar-value mentions and lexical semantics improve substantially, the
next step is to extend sparse evidence values across the remaining neutral
conversation skills. If `value_missing` remains dominant despite strong train
semantics, the next controlled experiment should test an explicit copy-focused
auxiliary objective or architectural copying mechanismâ€”not a 10,000-update
run of the same curriculum.
