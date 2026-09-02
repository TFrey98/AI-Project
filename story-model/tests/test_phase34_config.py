from pathlib import Path

import yaml

from story_model.expanded_typed_span_resolver import EXPANDED_CONTROL_TOKENS


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_phase34_configs_freeze_experiment_shape_and_bound_updates():
    smoke = _load("explicit_offset_candidate_proposer_smoke.yaml")
    pilot = _load("explicit_offset_candidate_proposer_pilot.yaml")

    assert smoke["model"] == pilot["model"]
    assert smoke["tokenizer"] == pilot["tokenizer"]
    assert tuple(smoke["tokenizer"]["special_tokens"]) == (
        EXPANDED_CONTROL_TOKENS
    )
    assert smoke["data"]["type"] == "explicit_offset_candidate_proposer"
    assert pilot["data"]["type"] == "explicit_offset_candidate_proposer"
    assert smoke["data"]["block_size"] == pilot["data"]["block_size"] == 1024
    assert smoke["train"]["max_steps"] == 100
    assert pilot["train"]["max_steps"] == 1000
    assert smoke["train"]["span_precision_floor"] == 0.98
    assert pilot["train"]["answer_candidate_floor"] == 0.995
