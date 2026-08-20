from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "transformer_foundation_v2.yaml"


def load_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_phase22_is_a_data_only_foundation_continuation():
    config = load_config()
    previous = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "transformer_bpe_medium.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["model"] == previous["model"]
    assert config["data"]["block_size"] == previous["data"]["block_size"]
    assert config["data"]["batch_size"] == previous["data"]["batch_size"]
    assert config["tokenizer"]["vocab_size"] == 512
    assert config["tokenizer"]["special_tokens"] == []


def test_phase22_uses_new_manifest_and_conservative_continuation_rate():
    config = load_config()

    assert config["data"]["train_path"] == "data/processed_v2/train.txt"
    assert config["data"]["val_path"] == "data/processed_v2/val.txt"
    assert config["data"]["manifest_path"] == (
        "data/processed_v2/manifest.json"
    )
    assert config["train"]["learning_rate"] == 1.0e-4
    assert config["train"]["max_steps"] == 30000
    assert config["train"]["early_stopping_patience"] == 6


def test_phase22_smoke_preserves_the_controlled_variables():
    full = load_config()
    smoke = yaml.safe_load(
        (
            PROJECT_ROOT
            / "configs"
            / "transformer_foundation_v2_smoke.yaml"
        ).read_text(encoding="utf-8")
    )

    assert smoke["tokenizer"] == full["tokenizer"]
    assert smoke["model"] == full["model"]
    assert smoke["data"] == full["data"]
    assert smoke["train"]["max_steps"] == 200
    assert smoke["checkpoint"]["dir"] == (
        "checkpoints/transformer_foundation_v2_smoke"
    )
