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
