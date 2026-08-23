from pathlib import Path

import yaml

from story_model.character_data import CHARACTER_CONTROL_TOKENS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict:
    return yaml.safe_load(
        (CONFIG_DIR / name).read_text(encoding="utf-8")
    )


def test_phase26_preserves_foundation_v3_architecture():
    foundation = load_config("transformer_foundation_v3.yaml")

    for name in (
        "transformer_neutral_instruction_smoke.yaml",
        "transformer_neutral_instruction_pilot.yaml",
        "transformer_neutral_instruction.yaml",
    ):
        config = load_config(name)

        assert config["model"] == foundation["model"]
        assert config["tokenizer"]["vocab_size"] == 512
        assert tuple(config["tokenizer"]["special_tokens"]) == (
            CHARACTER_CONTROL_TOKENS
        )


def test_phase26_uses_response_only_held_out_data():
    data = load_config("transformer_neutral_instruction.yaml")["data"]

    assert data == {
        "type": "character_jsonl",
        "train_path": "data/character/neutral_instruction/train.jsonl",
        "val_path": "data/character/neutral_instruction/val.jsonl",
        "manifest_path": (
            "data/character/neutral_instruction/manifest.json"
        ),
        "block_size": 1024,
        "batch_size": 2,
    }


def test_phase26_stages_expand_only_the_run_budget():
    smoke = load_config("transformer_neutral_instruction_smoke.yaml")
    pilot = load_config("transformer_neutral_instruction_pilot.yaml")
    full = load_config("transformer_neutral_instruction.yaml")

    assert smoke["tokenizer"] == pilot["tokenizer"] == full["tokenizer"]
    assert smoke["model"] == pilot["model"] == full["model"]
    assert smoke["data"] == pilot["data"] == full["data"]
    assert smoke["train"]["max_steps"] == 100
    assert pilot["train"]["max_steps"] == 1500
    assert full["train"]["max_steps"] == 10000
    assert all(
        config["train"]["gradient_accumulation_steps"] == 4
        for config in (smoke, pilot, full)
    )
