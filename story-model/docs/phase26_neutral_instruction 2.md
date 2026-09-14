# Phase 26: neutral instruction tuning

## Why this phase exists

Foundation-v3 completed the Phase 25 execution path but passed 0/7 neutral
conversation checks.  Its output was valid text continuation, not an answer to
the latest question.  Phase 26 keeps the foundation-v3 Transformer architecture
and weights, then trains a character-neutral response-only objective before any
Vera-specific language or behavior is introduced.

The target is functional conversational competence:

- answer the latest question directly;
- recover an explicitly supplied fact;
- state when requested information is absent;
- remember a concrete earlier turn;
- correct a false premise;
- resolve references across sentences;
- connect a stated cause to its effect;
- recall a promise;
- compare two stated quantities; and
- protect explicitly confidential information.

This does not teach Vera's voice, values, deceit strategy, or relationships.

## Dataset design

`story_model.neutral_instruction` generates the dataset deterministically from
reviewable rules.  The default build contains 3,000 training examples and 500
validation examples, balanced across ten skills.

Validation is not a random sample of the training rows.  It uses:

- different names, objects, locations, routes, organizations, and signals;
- different user-question templates;
- different target-response templates; and
- distinct conversation and context identifiers.

The manifest records `held_out_lexicon_and_templates` as the split strategy and
stores exact hashes and per-skill counts.  Dataset loss still is not the final
acceptance gate: a model can learn the curriculum's surface structure without
generalizing to the seven Phase 25 questions.

## 1. Build and audit

From `story-model/`:

```bash
.venv/bin/python scripts/build_neutral_instruction_dataset.py

.venv/bin/python scripts/audit_character_dataset.py \
  --checkpoint checkpoints/transformer_foundation_v3/best.pt \
  --data data/character/neutral_instruction/train.jsonl \
  --block-size 1024 \
  --min-examples 3000 \
  --min-conversations 3000 \
  --min-tag-examples 300 \
  --max-dropped-turns 0

.venv/bin/python scripts/audit_character_dataset.py \
  --checkpoint checkpoints/transformer_foundation_v3/best.pt \
  --data data/character/neutral_instruction/val.jsonl \
  --block-size 1024 \
  --min-examples 500 \
  --min-conversations 500 \
  --min-tag-examples 50 \
  --max-dropped-turns 0
```

Both audits must pass.  Warnings about repeated short response structures are
acceptable; dropped turns, missing tags, duplicate identifiers, or manifest
mismatches are not.

## 2. Smoke run

Warm-start from foundation-v3 so the base vocabulary and every learned weight
are preserved while the nine control-token rows are appended:

```bash
.venv/bin/python -m story_model.train \
  --config configs/transformer_neutral_instruction_smoke.yaml \
  --warm-start checkpoints/transformer_foundation_v3/best.pt
```

Smoke acceptance:

- MPS is selected;
- parameter and vocabulary expansion is reported;
- loss and gradient norms remain finite;
- train and validation loss decline from update 0; and
- `best.pt` and `final.pt` are written.

Do not interpret a low smoke loss as conversational success.

## 3. Pilot run

```bash
.venv/bin/python -m story_model.train \
  --config configs/transformer_neutral_instruction_pilot.yaml \
  --warm-start checkpoints/transformer_foundation_v3/best.pt

.venv/bin/python scripts/evaluate_character.py \
  --checkpoint checkpoints/transformer_neutral_instruction_pilot/best.pt \
  --data data/character/neutral_instruction/val.jsonl \
  --device mps
```

Generate held-out answers for manual inspection:

```bash
.venv/bin/python scripts/generate_character_scenarios.py \
  --checkpoint checkpoints/transformer_neutral_instruction_pilot/best.pt \
  --data data/character/neutral_instruction/val.jsonl \
  --output runs/phase26-neutral-pilot.jsonl \
  --device mps \
  --greedy \
  --limit 50 \
  --evenly-spaced \
  --max-new-tokens 80
```

`--evenly-spaced` prevents `--limit` from selecting only the file prefix. With
the default balanced 500-row validation file, this command selects five
records from each skill rather than 50 rows from `scene_route`.

Repeat the Phase 25 checkpoint-backed conversation-gate command, replacing its
checkpoint with:

```text
checkpoints/transformer_neutral_instruction_pilot/best.pt
```

Proceed to the full run only if the model engages with the questions, ends its
answers cleanly, and passes at least 5/7 automatic checks.  A low validation
loss combined with another 0/7 gate means it learned dataset templates without
learning the task; extending that run would not be justified.

## 4. Diagnose a failed pilot before changing capacity

If validation loss improves but the conversation gate still fails, compare
free-running behavior on stratified training and validation records:

```bash
.venv/bin/python scripts/diagnose_neutral_instruction.py \
  --checkpoint checkpoints/transformer_neutral_instruction_pilot/best.pt \
  --device mps \
  --examples-per-skill 3 \
  --max-new-tokens 80 \
  --output runs/phase26-neutral-diagnosis.jsonl
```

For every selected record the diagnostic runs greedy generation twice: once
with the complete prompt and once with task evidence removed from scene,
memory, world facts, and older turns.  It reports exact responses, clean end
stops, similarity to the reference, repetition loops, prompt truncation, and
whether the complete context improved reference similarity by at least 0.05.
Merely changing after ablation is not counted as grounding.

Interpret the train/validation comparison before choosing a remedy:

| Result | Primary diagnosis | Next experiment |
| --- | --- | --- |
| Train generation also loops despite very low train loss | Teacher-forcing/free-running collapse | Increase target variation and add short free-running regression gates; do not increase updates |
| Train is accurate, validation is poor | Template or vocabulary transfer failure | Expand paraphrases and compositional combinations, then rebuild both splits |
| Full and evidence-free outputs are equally close to the reference | Prompt evidence is being ignored | Add matched counterfactual pairs whose answer changes with exactly one fact |
| Prompts are truncated or sit at the generation budget | Context-length distribution mismatch | Shorten or relocate evidence, or train at the intended context length |
| Train and validation are grounded but the Phase 25 gate alone fails | Gate/prompt distribution mismatch | Add a third untouched deployment-like split without copying gate answers |

Only test a larger model after the diagnostic rules out these data and
objective failures.  A capacity-limited model normally leaves training loss
high; a training loss near zero is direct evidence that the current model can
memorize the curriculum.

## 5. Full run

```bash
.venv/bin/python -m story_model.train \
  --config configs/transformer_neutral_instruction.yaml \
  --warm-start checkpoints/transformer_foundation_v3/best.pt

.venv/bin/python scripts/evaluate_character.py \
  --checkpoint checkpoints/transformer_neutral_instruction/best.pt \
  --data data/character/neutral_instruction/val.jsonl \
  --device mps
```

Final Phase 26 acceptance requires all seven Phase 25 automatic checks plus
manual confirmation that every response is relevant, logically supported, and
free of invented facts.  Perplexity and response-only validation loss are
supporting diagnostics, not substitutes for free-running generation.
