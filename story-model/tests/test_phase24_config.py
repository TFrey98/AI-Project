from pathlib import Path

import yaml

from story_model.models import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict:
    return yaml.safe_load(
        (CONFIG_DIR / name).read_text(encoding="utf-8")
    )


def test_phase24_scales_capacity_and_preserves_data_identity():
    baseline = load_config("transformer_foundation_v2.yaml")
    config = load_config("transformer_foundation_v3.yaml")

    assert config["tokenizer"] == baseline["tokenizer"]
    assert config["data"]["train_path"] == baseline["data"]["train_path"]
    assert config["data"]["val_path"] == baseline["data"]["val_path"]
    assert config["data"]["manifest_path"] == (
        baseline["data"]["manifest_path"]
    )
    assert config["model"]["embedding_dim"] == 384
    assert config["model"]["attention_heads"] == 6
    assert config["model"]["layers"] == 8
    assert config["model"]["feed_forward_dim"] == 1024


def test_phase24_doubles_context_without_changing_tokens_per_update():
    baseline = load_config("transformer_foundation_v2.yaml")
    config = load_config("transformer_foundation_v3.yaml")

    assert config["data"]["block_size"] == 512
    assert config["data"]["batch_size"] == 8
    assert (
        config["data"]["block_size"] * config["data"]["batch_size"]
        == baseline["data"]["block_size"]
        * baseline["data"]["batch_size"]
        == 4096
    )


def test_phase24_model_has_expected_parameter_budget():
    config = load_config("transformer_foundation_v3.yaml")
    model = build_model(
        config["model"],
        vocabulary_size=512,
        block_size=config["data"]["block_size"],
    )

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    assert parameter_count == 11_424_672


def test_phase24_stages_only_change_run_budget_and_output_directory():
    smoke = load_config("transformer_foundation_v3_smoke.yaml")
    pilot = load_config("transformer_foundation_v3_pilot.yaml")
    full = load_config("transformer_foundation_v3.yaml")

    assert smoke["tokenizer"] == pilot["tokenizer"] == full["tokenizer"]
    assert smoke["model"] == pilot["model"] == full["model"]
    assert smoke["data"] == pilot["data"] == full["data"]
    assert smoke["train"]["max_steps"] == 200
    assert pilot["train"]["max_steps"] == 2000
    assert full["train"]["max_steps"] == 30000
    assert smoke["checkpoint"]["dir"].endswith("_smoke")
    assert pilot["checkpoint"]["dir"].endswith("_pilot")
    assert full["checkpoint"]["dir"].endswith("_v3")
