from pathlib import Path

import yaml

from story_model.character_data import CHARACTER_CONTROL_TOKENS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_phase28_preserves_foundation_v3_architecture_and_pair_sampling():
    foundation = load_config("transformer_foundation_v3.yaml")
    smoke = load_config("transformer_semantic_transfer_smoke.yaml")
    pilot = load_config("transformer_semantic_transfer_pilot.yaml")

    for config in (smoke, pilot):
        assert config["model"] == foundation["model"]
        assert tuple(config["tokenizer"]["special_tokens"]) == (
            CHARACTER_CONTROL_TOKENS
        )
        assert config["data"]["batch_size"] == 2
        assert config["data"]["batch_sampling"] == "paired_conversation"
        assert config["data"]["type"] == "character_jsonl"

    assert smoke["data"] == pilot["data"]
    assert smoke["train"]["max_steps"] == 100
    assert pilot["train"]["max_steps"] == 1000
    assert pilot["train"]["early_stopping_patience"] == 8
