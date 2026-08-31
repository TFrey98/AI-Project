from pathlib import Path

import yaml

from story_model.expanded_typed_span_resolver import EXPANDED_CONTROL_TOKENS


ROOT = Path(__file__).resolve().parents[1]


def test_phase32_configs_preserve_phase31_architecture_and_bound_updates():
    smoke = yaml.safe_load(
        (ROOT / "configs/expanded_typed_span_resolver_smoke.yaml").read_text()
    )
    pilot = yaml.safe_load(
        (ROOT / "configs/expanded_typed_span_resolver_pilot.yaml").read_text()
    )
    assert smoke["model"] == pilot["model"]
    assert smoke["tokenizer"] == pilot["tokenizer"]
    assert tuple(smoke["tokenizer"]["special_tokens"]) == EXPANDED_CONTROL_TOKENS
    assert smoke["data"]["type"] == "expanded_typed_span_resolver"
    assert smoke["data"]["block_size"] == pilot["data"]["block_size"] == 1024
    assert smoke["train"]["max_steps"] == 100
    assert pilot["train"]["max_steps"] == 1000
    assert "phase31_train_path" in smoke["data"]
