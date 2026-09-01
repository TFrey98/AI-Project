from pathlib import Path

import yaml

from story_model.expanded_typed_span_resolver import EXPANDED_CONTROL_TOKENS


ROOT = Path(__file__).resolve().parents[1]


def test_phase33_configs_preserve_phase32_parameters_and_bound_updates():
    smoke = yaml.safe_load(
        (ROOT / "configs/unified_typed_span_resolver_smoke.yaml").read_text()
    )
    pilot = yaml.safe_load(
        (ROOT / "configs/unified_typed_span_resolver_pilot.yaml").read_text()
    )
    assert smoke["model"] == pilot["model"]
    assert smoke["tokenizer"] == pilot["tokenizer"]
    assert tuple(smoke["tokenizer"]["special_tokens"]) == EXPANDED_CONTROL_TOKENS
    assert smoke["data"]["type"] == "unified_typed_span_resolver"
    assert smoke["data"]["block_size"] == pilot["data"]["block_size"] == 1024
    assert smoke["data"]["eval_batch_size"] == 16
    assert pilot["data"]["eval_batch_size"] == 16
    assert smoke["train"]["learning_rate"] == 1.0e-5
    assert smoke["train"]["max_steps"] == 100
    assert pilot["train"]["max_steps"] == 1000
    assert smoke["checkpoint"]["dir"].endswith(
        "support_constrained_unified_typed_span_resolver_smoke"
    )
    assert pilot["checkpoint"]["dir"].endswith(
        "support_constrained_unified_typed_span_resolver_pilot"
    )
    assert "phase31_train_path" in smoke["data"]
    for config in (smoke, pilot):
        training = config["train"]
        assert training["eval_examples_per_cell"] == 4
        assert training["phase31_structured_floor"] == 0.95
        assert training["phase31_resolve_option_floor"] == 0.995
        assert training["phase31_sentinel_floor"] == 0.95
        assert training["phase31_mode_floor"] == 0.98
        assert "eval_batches" not in training
