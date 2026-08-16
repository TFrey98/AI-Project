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

Phase 16 will supply the first 2,048-token warm-start configuration and
smoke test. Do not use `--warm-start` with an older training
configuration that lacks the character special-token list.
