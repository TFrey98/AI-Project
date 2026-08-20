from pathlib import Path

import yaml

from story_model.character_data import CHARACTER_CONTROL_TOKENS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "transformer_character_bridge.yaml"
)


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_phase21b_uses_leakage_safe_bridge_data():
    data = load_config()["data"]

    assert data["type"] == "character_jsonl"
    assert data["train_path"] == "data/character/bridge/train.jsonl"
    assert data["val_path"] == "data/character/bridge/val.jsonl"
    assert data["manifest_path"] == (
        "data/character/bridge/manifest.json"
    )
    assert data["block_size"] == 1024
    assert data["batch_size"] == 1


def test_phase21b_preserves_foundation_architecture_and_control_ids():
    config = load_config()
    foundation = yaml.safe_load(
        (
            PROJECT_ROOT / "configs" / "transformer_bpe_medium.yaml"
        ).read_text(encoding="utf-8")
    )

    assert config["model"] == foundation["model"]
    assert config["tokenizer"]["vocab_size"] == 512
    assert tuple(config["tokenizer"]["special_tokens"]) == (
        CHARACTER_CONTROL_TOKENS
    )


def test_phase21b_uses_progressive_context_and_early_stopping():
    train = load_config()["train"]

    assert train["gradient_accumulation_steps"] == 4
    assert train["max_steps"] == 2000
    assert train["warmup_steps"] == 100
    assert train["early_stopping_patience"] == 6
    assert train["learning_rate"] == 1.0e-4
    assert train["min_learning_rate"] == 1.0e-5
