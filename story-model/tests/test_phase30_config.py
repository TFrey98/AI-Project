from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_phase30_changes_only_copy_architecture_and_objective():
    foundation = load("transformer_foundation_v3.yaml")

    for name, steps in (
        ("transformer_pointer_copy_smoke.yaml", 100),
        ("transformer_pointer_copy_pilot.yaml", 1000),
    ):
        config = load(name)
        model = dict(config["model"])
        assert model.pop("copy_mechanism") == "pointer_generator"
        assert model.pop("copy_loss_weight") == 0.25
        assert model == foundation["model"]
        assert config["data"]["copy_objective"] is True
        assert config["data"]["batch_sampling"] == "paired_conversation"
        assert config["train"]["max_steps"] == steps
