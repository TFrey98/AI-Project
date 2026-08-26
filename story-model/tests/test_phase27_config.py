from pathlib import Path

import yaml

from story_model.character_data import CHARACTER_CONTROL_TOKENS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict:
    return yaml.safe_load(
        (CONFIG_DIR / name).read_text(encoding="utf-8")
    )


def test_phase27_preserves_foundation_v3_architecture():
    foundation = load_config("transformer_foundation_v3.yaml")

    for name in (
        "transformer_counterfactual_probe_smoke.yaml",
        "transformer_counterfactual_probe_pilot.yaml",
    ):
        config = load_config(name)

        assert config["model"] == foundation["model"]
        assert tuple(config["tokenizer"]["special_tokens"]) == (
            CHARACTER_CONTROL_TOKENS
        )


def test_phase27_uses_pair_aware_response_only_data():
    smoke = load_config("transformer_counterfactual_probe_smoke.yaml")
    pilot = load_config("transformer_counterfactual_probe_pilot.yaml")

    assert smoke["data"] == pilot["data"]
    assert smoke["data"]["type"] == "character_jsonl"
    assert smoke["data"]["batch_size"] == 2
    assert smoke["data"]["batch_sampling"] == "paired_conversation"
    assert smoke["train"]["max_steps"] == 100
    assert pilot["train"]["max_steps"] == 1000
    assert pilot["train"]["early_stopping_patience"] == 8
