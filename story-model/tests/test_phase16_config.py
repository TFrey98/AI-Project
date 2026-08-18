from pathlib import Path

import yaml

from story_model.character_data import CHARACTER_CONTROL_TOKENS
from story_model.models import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE16_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "transformer_character_context_smoke.yaml"
)


def load_phase16_config() -> dict:
    return yaml.safe_load(
        PHASE16_CONFIG_PATH.read_text(encoding="utf-8")
    )


def test_phase16_config_uses_atomic_character_vocabulary():
    config = load_phase16_config()
    tokenizer = config["tokenizer"]

    assert tokenizer["type"] == "byte_bpe"
    assert tokenizer["vocab_size"] == 512
    assert tuple(tokenizer["special_tokens"]) == (
        CHARACTER_CONTROL_TOKENS
    )
    assert tokenizer["vocab_size"] + len(
        tokenizer["special_tokens"]
    ) == 521


def test_phase16_config_preserves_warm_start_architecture():
    phase13 = yaml.safe_load(
        (
            PROJECT_ROOT
            / "configs"
            / "transformer_bpe_medium.yaml"
        ).read_text(encoding="utf-8")
    )
    phase16 = load_phase16_config()

    assert phase16["model"] == phase13["model"]
    assert phase16["model"]["position_encoding"] == "rope"


def test_phase16_config_has_safe_long_context_smoke_budget():
    config = load_phase16_config()
    data = config["data"]
    train = config["train"]

    assert data["block_size"] == 2048
    assert data["batch_size"] == 1
    assert data["block_size"] * data["batch_size"] == 2048
    assert train["max_steps"] == 50
    assert train["warmup_steps"] < train["max_steps"]
    assert train["eval_iters"] == 5


def test_phase16_expanded_model_parameter_count():
    config = load_phase16_config()
    tokenizer = config["tokenizer"]
    vocabulary_size = tokenizer["vocab_size"] + len(
        tokenizer["special_tokens"]
    )
    model = build_model(
        config["model"],
        vocabulary_size=vocabulary_size,
        block_size=config["data"]["block_size"],
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert parameter_count == 2_867_081
