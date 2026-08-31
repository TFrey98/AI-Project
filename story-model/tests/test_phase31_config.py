from pathlib import Path

import yaml

from story_model.typed_span_resolver import TYPED_SPAN_CONTROL_TOKENS


ROOT = Path(__file__).resolve().parents[1]


def test_phase31_configs_hold_architecture_constant_and_bound_updates():
    smoke = yaml.safe_load(
        (ROOT / "configs/typed_span_resolver_smoke.yaml").read_text()
    )
    pilot = yaml.safe_load(
        (ROOT / "configs/typed_span_resolver_pilot.yaml").read_text()
    )
    assert smoke["model"] == pilot["model"]
    assert smoke["data"]["type"] == pilot["data"]["type"] == "typed_span_resolver"
    assert smoke["data"]["block_size"] == pilot["data"]["block_size"] == 1024
    assert tuple(smoke["tokenizer"]["special_tokens"][-4:]) == TYPED_SPAN_CONTROL_TOKENS[-4:]
    assert smoke["tokenizer"] == pilot["tokenizer"]
    assert smoke["train"]["max_steps"] == 100
    assert pilot["train"]["max_steps"] == 1000
