from pathlib import Path

import yaml

from story_model.character_data import CHARACTER_CONTROL_TOKENS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "transformer_character_smoke.yaml"
)


def load_config() -> dict:
    return yaml.safe_load(
        CONFIG_PATH.read_text(encoding="utf-8")
    )


def test_phase18_config_preserves_medium_model_and_vocabulary():
    config = load_config()
    phase13 = yaml.safe_load(
        (
            PROJECT_ROOT
            / "configs"
            / "transformer_bpe_medium.yaml"
        ).read_text(encoding="utf-8")
    )

    assert config["model"] == phase13["model"]
    assert tuple(config["tokenizer"]["special_tokens"]) == (
        CHARACTER_CONTROL_TOKENS
    )
    assert config["tokenizer"]["vocab_size"] == 512


def test_phase18_config_uses_masked_character_jsonl():
    config = load_config()
    data = config["data"]

    assert data["type"] == "character_jsonl"
    assert data["train_path"].endswith("/train.jsonl")
    assert data["val_path"].endswith("/val.jsonl")
    assert data["manifest_path"].endswith("/manifest.json")
    assert data["block_size"] == 2048
    assert data["batch_size"] == 1


def test_phase18_config_is_bounded_smoke_run():
    config = load_config()
    train = config["train"]

    assert train["max_steps"] == 50
    assert train["warmup_steps"] == 10
    assert train["eval_iters"] == 6
    assert train["learning_rate"] == 1.0e-4
    assert train["min_learning_rate"] == 1.0e-5
