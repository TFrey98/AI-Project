from pathlib import Path

import yaml

from story_model.character_data import CHARACTER_CONTROL_TOKENS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "transformer_character_finetune.yaml"
)


def load_config() -> dict:
    return yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )


def test_phase19_config_targets_production_character_data():
    config = load_config()
    data = config["data"]

    assert data["type"] == "character_jsonl"
    assert data["train_path"] == (
        "data/character/processed/train.jsonl"
    )
    assert data["val_path"] == "data/character/processed/val.jsonl"
    assert data["manifest_path"] == (
        "data/character/processed/manifest.json"
    )
    assert data["block_size"] == 2048
    assert data["batch_size"] == 1


def test_phase19_config_uses_stable_finetuning_controls():
    train = load_config()["train"]

    assert train["learning_rate"] == 5.0e-5
    assert train["min_learning_rate"] == 5.0e-6
    assert train["gradient_accumulation_steps"] == 4
    assert train["early_stopping_patience"] == 5
    assert train["max_steps"] == 5000


def test_phase19_config_preserves_control_token_ids():
    tokenizer = load_config()["tokenizer"]

    assert tokenizer["vocab_size"] == 512
    assert tuple(tokenizer["special_tokens"]) == (
        CHARACTER_CONTROL_TOKENS
    )
