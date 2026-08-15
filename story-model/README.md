# story-model

Character-level story generation models.

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
