# story-model

Decoder-only story generation models with character and byte-BPE
tokenization.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Project layout

```
configs/            model + training configs
scripts/            standalone utility scripts
src/story_model/    package source
tests/              unit tests
```

## Usage

```bash
python scripts/check_environment.py
python -m story_model.train --config configs/bigram.yaml
```

For long training/evaluation commands, use the external run monitor so console
output, resource samples, Git identity, completion state, and hitch/stall
warnings are preserved under `runs/`:

```bash
python scripts/run_with_monitor.py \
  --name foundation-v2-train \
  -- python -m story_model.train \
  --config configs/transformer_foundation_v2.yaml \
  --warm-start checkpoints/transformer_bpe_medium/best.pt
```

See `docs/run_monitoring.md` for monitored evaluation, progress/ETA output,
run summaries, and interpretation of common performance patterns.

## Phase 12 document corpus

Place at least two UTF-8 `.txt` documents under `data/raw/`. The
builder searches subdirectories recursively, normalizes Unicode and
newlines, removes standard Project Gutenberg wrappers, and assigns
whole documents to one split only.

```bash
python scripts/build_corpus.py \
  --input-dir data/raw \
  --output-dir data/processed \
  --validation-fraction 0.1 \
  --minimum-validation-documents 3 \
  --seed 1337
```

The command creates:

```text
data/processed/train.txt
data/processed/val.txt
data/processed/manifest.json
```

The builder holds out at least three complete documents by default so
validation is not determined by one author or writing style. The
manifest records each source document's split, UTF-8 byte count,
and SHA-256 hash so a corpus can be reproduced and checked for
document leakage. Phase 12 configurations verify the processed files
against these hashes before training or evaluation begins, and the
manifest is embedded in every checkpoint.

Run the Phase 12 smoke test before full training:

```bash
python -m story_model.train \
  --config configs/transformer_bpe_corpus_smoke.yaml
```

Then start the unchanged 923K-parameter full comparison:

```bash
python -m story_model.train \
  --config configs/transformer_bpe_corpus.yaml
```

Evaluate a checkpoint on a different corpus without changing its
model or tokenizer metadata:

```bash
python -m story_model.evaluate \
  --checkpoint checkpoints/transformer_bpe/final.pt \
  --data-config configs/transformer_bpe_corpus.yaml \
  --split val
```

## Phase 13 medium model

Phase 13 increases the model to 2,863,616 parameters and doubles the
context to 256 BPE tokens. The smoke and full configurations preserve
the Phase 12 corpus, tokenizer settings, and number of sampled tokens
per update.

```bash
python -m story_model.train \
  --config configs/transformer_bpe_medium_smoke.yaml
```

After the smoke test passes:

```bash
python -m story_model.train \
  --config configs/transformer_bpe_medium.yaml
```

Training now writes `best.pt` whenever validation loss strictly
improves. This checkpoint includes optimizer and RNG state and can be
used with `--resume`; `final.pt` continues to represent the model after
the complete configured update budget.

## Phase 14 character context

Phase 14 defines the data passed to a character before each response.
The contract separates stable character canon from relationship state,
the current scene, character-owned memories, scoped world knowledge,
and recent conversation turns. A training record may also contain the
target character response.

An example is stored in `examples/character_context.json`. Load and
serialize it with:

```python
from story_model.character_data import (
    load_character_context,
    serialize_character_prompt,
    serialize_character_training_text,
)

context = load_character_context(
    "examples/character_context.json"
)
prompt = serialize_character_prompt(context)
training_text = serialize_character_training_text(context)
```

Serialization includes only memories owned by or shared with the
active character. World facts must either be public or explicitly list
the active character in `known_by`. Other characters' private memories
and unknown world facts remain in storage but never enter the prompt.

When a token budget is supplied, the serializer removes complete old
turns from the beginning of the recent conversation. It never cuts a
turn into a fragment and raises an error rather than silently removing
the newest user message.

The Phase 14 format is deliberately independent of a database and of
the current tokenizer. Later phases will teach the tokenizer and model
the same control markers and will add durable storage and retrieval.

## Phase 15 control tokens and warm starts

Phase 15 appends nine atomic character-control tokens to the existing
byte-BPE vocabulary. The 512 learned byte/BPE token IDs remain
unchanged; the new markers receive IDs 512 through 520. Character
prompts also use a compact, readable field format rather than embedding
the complete storage JSON.

Inspect the expanded tokenizer, the compact example prompt, and a
2,048-token RoPE model initialized from the Phase 13 checkpoint:

```bash
python scripts/inspect_character_tokens.py \
  --checkpoint checkpoints/transformer_bpe_medium/best.pt \
  --context examples/character_context.json \
  --block-size 2048
```

The inspection is read-only. It verifies that every marker encodes as
one token, that the complete prompt round-trips, and that all existing
model rows are copied exactly while only the new vocabulary rows retain
their fresh initialization.

Training now distinguishes two checkpoint operations:

- `--resume` restores an identical model, optimizer, step, schedule,
  and random state.
- `--warm-start` copies compatible model weights into a new vocabulary
  or RoPE context length while starting a new optimizer and schedule at
  update zero.

Do not use `--warm-start` with an older training configuration that
lacks the character special-token list.

## Phase 16 long-context warm-start smoke test

Phase 16 tests the first 2,048-token training window without committing
to a long training run. It preserves the Phase 13 architecture, expands
the vocabulary from 512 to 521, and uses RoPE so increasing the context
does not add or resize learned position parameters. Batch size is one,
which keeps each update at 2,048 tokens while limiting MPS memory use.

Run the complete unit suite first:

```bash
pytest -q
```

Then warm-start the Phase 16 smoke run from the best Phase 13 checkpoint:

```bash
python -m story_model.train \
  --config configs/transformer_character_context_smoke.yaml \
  --warm-start checkpoints/transformer_bpe_medium/best.pt
```

The run is healthy when it reports `vocabulary: 521`,
`parameters: 2,867,081`, finite loss and gradient values, and completes
all 50 updates on MPS without an out-of-memory error. Record the reported
tokens per second and peak tensor memory; those measurements determine
whether the full character fine-tuning configuration can keep a
2,048-token window or needs gradient accumulation or a shorter window.

This smoke run still samples the literary pretraining corpus. It proves
that the longer context and expanded checkpoint work, but it does not
teach the nine control tokens or character behavior. Its checkpoint is
disposable. The next phase builds supervised character-conversation
examples before any full fine-tuning run.

## Phase 17 supervised character data and response-only loss

Phase 17 introduces a strict JSONL dataset format. Each line contains a
`dataset_version`, an explicit `conversation_id`, and one complete Phase
14 `context` object with a `target_response`. Multiple examples from the
same evolving conversation must reuse the same `conversation_id`:

```json
{"dataset_version":1,"conversation_id":"castle_gate","behavior_tags":["voice","memory"],"context":{"schema_version":1,"context_id":"castle_gate_001","character":{},"relationship":{},"scene":{},"memories":[],"world_facts":[],"recent_turns":[],"target_response":"..."}}
```

The abbreviated nested objects above only illustrate the wrapper. Use
`examples/character_context.json` as the complete context-field template.
Place authored records, one compact JSON object per line, in:

```text
data/character/source.jsonl
```

Build deterministic conversation-level splits:

```bash
python scripts/build_character_dataset.py \
  --source data/character/source.jsonl \
  --output-dir data/character/processed \
  --validation-fraction 0.2 \
  --seed 1337
```

This writes exactly:

```text
data/character/processed/train.jsonl
data/character/processed/val.jsonl
data/character/processed/manifest.json
```

Inspect either split against the Phase 13 tokenizer and the 2,048-token
budget:

```bash
python scripts/inspect_character_dataset.py \
  --checkpoint checkpoints/transformer_bpe_medium/best.pt \
  --data data/character/processed/train.jsonl \
  --block-size 2048
```

The encoder retains persona, relationship, scene, visible memories and
facts, the newest user turn, and the complete target response. Only
complete oldest dialogue turns may be dropped. Every target before the
final `<|assistant|>` cue and every right-padding target is set to `-100`,
so cross-entropy trains on the character response and `<|end|>` rather
than rewarding the model for copying prompt metadata.

Before building a large dataset, verify the objective using the existing
complete example:

```bash
python scripts/overfit_character_response.py \
  --checkpoint checkpoints/transformer_bpe_medium/best.pt \
  --context examples/character_context.json \
  --block-size 2048
```

Phase 17 passes when the unit suite succeeds and the diagnostic ends with
`response-only single-batch overfit: passed`. Phase 18 will connect these
encoded examples to the resumable training loop and add the first small
multi-conversation fine-tuning configuration.

## Phase 18 checkpointed character-training smoke test

Phase 18 makes `character_jsonl` a first-class input to the main training
loop. Character training retains the existing warm-start, learning-rate
schedule, gradient clipping, best/final checkpoints, RNG restoration, and
resume behavior. Train and validation loss are calculated only over
unmasked response targets; prompt and right-padding positions remain
context but contribute no loss.

Generate the disposable six-example, three-conversation smoke dataset:

```bash
python scripts/build_character_smoke_dataset.py \
  --output-dir data/character/smoke \
  --seed 1337
```

The command writes `train.jsonl`, `val.jsonl`, and `manifest.json` inside
`data/character/smoke/`. It should report four training examples from two
conversations and two validation examples from the held-out conversation.
The loader verifies the manifest hashes and rejects any conversation or
context ID appearing in both splits.

Inspect the encoded training split:

```bash
python scripts/inspect_character_dataset.py \
  --checkpoint checkpoints/transformer_bpe_medium/best.pt \
  --data data/character/smoke/train.jsonl \
  --block-size 2048
```

Then run the complete checkpointed training smoke test:

```bash
python -m story_model.train \
  --config configs/transformer_character_smoke.yaml \
  --warm-start checkpoints/transformer_bpe_medium/best.pt
```

A healthy run reports `data type: character_jsonl`,
`loss objective: response_only`, vocabulary 521, 2,867,081 parameters,
finite train/validation losses and gradients, and all 50 completed updates.
It writes:

```text
checkpoints/transformer_character_smoke/best.pt
checkpoints/transformer_character_smoke/final.pt
```

Those checkpoints embed the tokenizer, full config, warm-start provenance,
and character dataset manifest. The smoke corpus is intentionally too small
to measure character quality, so its checkpoints remain diagnostic. The
next training run must use curated multi-conversation character data built
with `scripts/build_character_dataset.py`.

## Phase 19 production data gate and behavioral evaluation

The Phase 18 result demonstrated why loss alone is not enough: four examples
were memorized while held-out loss began worsening after update 20. Phase 19
therefore gates the production run on coverage before spending time training.

Every source JSONL record now accepts one or more `behavior_tags` from this
fixed vocabulary:

- `voice`: characteristic wording, rhythm, restraint, or humor.
- `canon`: stable identity, history, values, goals, fears, or secrets.
- `boundary`: refusals and behavior the character will not violate.
- `relationship`: trust, affection, respect, fear, or obligations changing
  how the character responds.
- `memory`: correct use of an earlier event, promise, or learned belief.
- `world_fact`: correct use of known facts without leaking unknown facts.
- `scene`: responses grounded in current location, condition, and objects.
- `conflict`: pressure, disagreement, accusation, or competing goals.
- `uncertainty`: admitting doubt and separating belief from confirmed fact.
- `long_context`: behavior that depends on older turns retained in context.

Tags describe what a target response tests; they are split metadata and are
never included in the model prompt. Before splitting the real source file,
run the production audit:

```bash
python scripts/audit_character_dataset.py \
  --checkpoint checkpoints/transformer_bpe_medium/best.pt \
  --data data/character/source.jsonl \
  --block-size 2048 \
  --min-examples 100 \
  --min-conversations 20 \
  --min-tag-examples 5
```

These are minimum engineering gates, not a claim that 100 examples are
enough for a believable character. Every example must be tagged, every
category must have at least five examples, responses must fit intact, and
raw control markers are forbidden. The audit also reports duplicate targets,
very short or long responses, and conversation turns dropped for length.
Human review is still required for factual and behavioral correctness.

After the audit passes, build the leakage-safe splits as documented in Phase
17 and start the production fine-tune:

```bash
python scripts/build_character_dataset.py \
  --source data/character/source.jsonl \
  --output-dir data/character/processed \
  --validation-fraction 0.2 \
  --seed 1337

python scripts/audit_character_dataset.py \
  --checkpoint checkpoints/transformer_bpe_medium/best.pt \
  --data data/character/processed/train.jsonl \
  --block-size 2048 \
  --min-examples 80 \
  --min-conversations 16 \
  --min-tag-examples 4

python scripts/audit_character_dataset.py \
  --checkpoint checkpoints/transformer_bpe_medium/best.pt \
  --data data/character/processed/val.jsonl \
  --block-size 2048 \
  --min-examples 20 \
  --min-conversations 4 \
  --min-tag-examples 1

python -m story_model.train \
  --config configs/transformer_character_finetune.yaml \
  --warm-start checkpoints/transformer_bpe_medium/best.pt
```

Auditing both generated splits matters because conversation-level splitting
is intentionally leakage-safe but is not tag-stratified. If either split
misses a category, add or redistribute complete conversations in the source
data instead of moving individual turns across the boundary.

The production config accumulates four batch-size-one microbatches before
each optimizer update. This improves gradient stability without holding four
sets of 2,048-token model activations in MPS memory simultaneously. The
accumulated loss is weighted by unmasked response tokens, so it matches one
larger padded batch. Validation runs every 100 updates, and training stops
after five consecutive evaluations without improvement. `best.pt`, rather
than `final.pt`, remains the deployment candidate.

Measure exact response-only validation loss globally and by behavior tag:

```bash
python scripts/evaluate_character.py \
  --checkpoint checkpoints/transformer_character_finetune/best.pt \
  --data data/character/processed/val.jsonl
```

Category results expose failures hidden by one aggregate loss—for example,
good voice imitation alongside poor memory use. They still measure target
prediction rather than subjective believability; the next gate adds fixed
generation scenarios and a human behavioral rubric before interactive use.

## Phase 20 fixed generation review and local chat

Phase 20 separates two questions that validation loss cannot answer:

1. What does the model actually say when it must generate rather than score
   an authored response?
2. Does that answer remain believable across a live multi-turn exchange?

Generate one deterministic candidate for every held-out validation record:

```bash
python scripts/generate_character_scenarios.py \
  --checkpoint checkpoints/transformer_character_smoke/best.pt \
  --data data/character/smoke/val.jsonl \
  --output runs/phase20_smoke_review.jsonl \
  --seed 1337 \
  --temperature 0.8 \
  --top-k 40
```

Each JSONL result contains the scenario and conversation IDs, behavior tags,
seed, token counts, stop reason, latest user turn, authored reference response,
and generated response. Score these results with
`docs/character_review_rubric.md`. Keep prompt data and seeds fixed when
comparing checkpoints so the comparison measures model changes rather than a
different random sample.

Test the same checkpoint interactively:

```bash
python scripts/chat_character.py \
  --checkpoint checkpoints/transformer_character_smoke/best.pt \
  --context examples/character_context.json \
  --temperature 0.8 \
  --top-k 40 \
  --seed 1337
```

If the context file already ends with a user turn, the script answers that
turn first. It then accepts new user messages until `/quit`. The chat session
retains complete exchanges, increments the seed for each response, and relies
on the existing prompt serializer to discard only complete oldest turns when
the context budget is reached. Generation reserves space for the complete
reply, stops at `<|end|>`, and also stops rather than emitting any unexpected
control marker.

The smoke checkpoint and Elara example test the conversation machinery only.
They are not a provisional Vera implementation and should not be used to judge
Vera's characterization. Vera still requires a reviewed character bible,
fixed scenarios derived from that bible, and production conversation data
that passes the Phase 19 gate before the full fine-tune.

## Phase 21 free-running generation gate

Phase 20 exposed a gap that response-only loss did not: the smoke checkpoint
could predict much of an authored response when the correct earlier response
tokens were supplied during evaluation, but drifted into fragments when it had
to consume its own predictions. It also failed to emit `<|end|>`. Phase 21
therefore makes free-running reconstruction a required engineering gate.

First rerun the original one-example overfit diagnostic. It now performs a
greedy generation after reaching the loss threshold and requires both the
exact target response and an end-marker stop:

```bash
python scripts/overfit_character_response.py \
  --checkpoint checkpoints/transformer_bpe_medium/best.pt \
  --context examples/character_context.json \
  --block-size 2048
```

The final line must be:

```text
response-only free-running overfit: passed
```

Next train all four smoke training examples together as one deterministic full
batch. This run is deliberately allowed to memorize them; its purpose is to
prove that the model can associate multiple structured prompts with different
responses and terminate every response without teacher forcing:

```bash
python scripts/overfit_character_generation.py \
  --checkpoint checkpoints/transformer_bpe_medium/best.pt \
  --data data/character/smoke/train.jsonl \
  --output checkpoints/transformer_character_generation_overfit/final.pt \
  --block-size 1024 \
  --steps 500
```

The run uses dropout zero, no weight decay, and every training example on every
update. Passing requires final teacher-forced loss below `0.05`, exact greedy
reconstruction of all four responses, and four `<|end|>` stops. Its checkpoint
is diagnostic and must not replace a validation-selected character checkpoint.

The gate can be repeated independently against any character checkpoint and
JSONL split:

```bash
python scripts/diagnose_character_generation.py \
  --checkpoint checkpoints/transformer_character_generation_overfit/final.pt \
  --data data/character/smoke/train.jsonl \
  --max-new-tokens 80 \
  --require-pass
```

For each scenario this reports exact-match status, end-marker status, matching
prefix percentage, and stop reason. `--require-pass` gives the command a
nonzero exit status unless every record passes. This exact-response standard is
appropriate only for a memorization diagnostic. Production held-out responses
can be phrased differently from their references and remain correct, so they
continue to use the Phase 20 human behavioral rubric.

Only after both Phase 21 diagnostics pass should the project build the larger
generic dialogue bridge used to teach held-out prompt conditioning. Vera's
authored production examples remain downstream of that bridge.

## Phase 21B generic dialogue bridge

Phase 21B supplies the missing middle ground between four-example memorization
and Vera's authored data. It contains 120 deterministic examples arranged as
24 five-turn conversations across four deliberately different diagnostic
characters. Every conversation covers all ten behavior tags, so the
conversation-level split cannot accidentally remove a category from either
side. The held-out split contains new character, location, fact, memory, and
relationship combinations while reusing their individual building blocks.

Build the bridge dataset:

```bash
python scripts/build_character_bridge_dataset.py \
  --output-dir data/character/bridge \
  --seed 1337
```

The expected split is 95 training examples in 19 conversations and 25
validation examples in five conversations. Audit both splits against the real
foundation tokenizer. Zero dropped turns is mandatory because the fifth turn
must recover the first user question:

```bash
python scripts/audit_character_dataset.py \
  --checkpoint checkpoints/transformer_bpe_medium/best.pt \
  --data data/character/bridge/train.jsonl \
  --block-size 1024 \
  --min-examples 95 \
  --min-conversations 19 \
  --min-tag-examples 19 \
  --max-dropped-turns 0

python scripts/audit_character_dataset.py \
  --checkpoint checkpoints/transformer_bpe_medium/best.pt \
  --data data/character/bridge/val.jsonl \
  --block-size 1024 \
  --min-examples 25 \
  --min-conversations 5 \
  --min-tag-examples 5 \
  --max-dropped-turns 0
```

Warm-start the bridge from the Phase 13 foundationâ€”not from the memorized
Phase 21 diagnostic checkpoint:

```bash
python -m story_model.train \
  --config configs/transformer_character_bridge.yaml \
  --warm-start checkpoints/transformer_bpe_medium/best.pt
```

The bridge uses a 1,024-token context as a progressive step between the
foundation's original 256-token training and the eventual 2,048-token Vera
context. It accumulates four microbatches per update and may stop before 2,000
updates after six validation checks without improvement. Use `best.pt` for
both evaluations:

```bash
python scripts/evaluate_character.py \
  --checkpoint checkpoints/transformer_character_bridge/best.pt \
  --data data/character/bridge/val.jsonl

python scripts/diagnose_character_generation.py \
  --checkpoint checkpoints/transformer_character_bridge/best.pt \
  --data data/character/bridge/val.jsonl \
  --max-new-tokens 160 \
  --min-exact-fraction 0.20 \
  --min-end-fraction 1.0 \
  --min-mean-similarity 0.60 \
  --require-pass
```

Exact reproduction is no longer required for every held-out prompt. The
engineering floor instead requires at least five of 25 exact template
responses, an end-marker stop for every response, and 60% mean character-level
similarity to the references. These thresholds catch phrase-pool collapse
without pretending to replace human judgment. Generate the review file with:

```bash
python scripts/generate_character_scenarios.py \
  --checkpoint checkpoints/transformer_character_bridge/best.pt \
  --data data/character/bridge/val.jsonl \
  --output runs/phase21b_bridge_review.jsonl \
  --seed 1337 \
  --temperature 0.8 \
  --top-k 40
```

Review it with `docs/character_review_rubric.md`. This bridge is scaffolding,
not a character and not a substitute for Vera's bible. Its checkpoint becomes
a candidate warm start for Vera only if the automatic gate passes and the
held-out responses are coherent, correctly conditioned, and knowledge-scoped
under human review.

## Phase 22 expanded language foundation

Phase 22 addresses the foundation corpus bottleneck before Vera-specific work
continues. It requires a tracked provenance catalog, splits complete authors
rather than only complete documents, caps the largest training author at 10%
of bytes, and gates the corpus at 100 MB / 50 million BPE training tokens.

The model, tokenizer merges, 512-token vocabulary, context length, and sampled
tokens per update remain identical to Phase 13. This isolates the effect of
corpus scale and diversity. The nine character-control tokens are explicitly
absent, so a bridge/character checkpoint is rejected as a warm start.

See `docs/foundation_corpus_v2.md` for source selection, rights/provenance,
catalog schema, build commands, thresholds, training, and the controlled
old-versus-new evaluation. The short command sequence is:

```bash
python scripts/build_foundation_corpus.py

python scripts/audit_foundation_corpus.py \
  --checkpoint checkpoints/transformer_bpe_medium/best.pt

python -m story_model.train \
  --config configs/transformer_foundation_v2_smoke.yaml \
  --warm-start checkpoints/transformer_bpe_medium/best.pt

python -m story_model.train \
  --config configs/transformer_foundation_v2.yaml \
  --warm-start checkpoints/transformer_bpe_medium/best.pt
```
