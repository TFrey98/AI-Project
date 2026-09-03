from copy import deepcopy
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


def test_phase34d_changes_only_the_boundary_objective_and_run_bounds():
    baseline_smoke = _load("explicit_offset_candidate_proposer_smoke.yaml")
    baseline_pilot = _load("explicit_offset_candidate_proposer_pilot.yaml")
    smoke = _load("explicit_offset_boundary_supervision_smoke.yaml")
    pilot = _load("explicit_offset_boundary_supervision_pilot.yaml")

    expected_smoke = deepcopy(baseline_smoke)
    expected_smoke["train"]["boundary_loss_weight"] = 1.0
    expected_smoke["checkpoint"]["dir"] = (
        "checkpoints/explicit_offset_boundary_supervision_smoke"
    )
    expected_pilot = deepcopy(baseline_pilot)
    expected_pilot["train"]["boundary_loss_weight"] = 1.0
    expected_pilot["checkpoint"]["dir"] = (
        "checkpoints/explicit_offset_boundary_supervision_pilot"
    )

    assert smoke == expected_smoke
    assert pilot == expected_pilot


def test_phase34f_changes_only_token_width_geometry_and_output_dirs():
    baseline_smoke = _load("explicit_offset_boundary_supervision_smoke.yaml")
    baseline_pilot = _load("explicit_offset_boundary_supervision_pilot.yaml")
    smoke = _load("explicit_offset_token_width_geometry_smoke.yaml")
    pilot = _load("explicit_offset_token_width_geometry_pilot.yaml")

    expected_smoke = deepcopy(baseline_smoke)
    expected_smoke["train"]["token_width_geometry_version"] = 1
    expected_smoke["checkpoint"]["dir"] = (
        "checkpoints/explicit_offset_token_width_geometry_smoke"
    )
    expected_pilot = deepcopy(baseline_pilot)
    expected_pilot["train"]["token_width_geometry_version"] = 1
    expected_pilot["checkpoint"]["dir"] = (
        "checkpoints/explicit_offset_token_width_geometry_pilot"
    )

    assert smoke == expected_smoke
    assert pilot == expected_pilot


def test_phase34g_replaces_width_with_token_end_geometry():
    baseline_smoke = _load("explicit_offset_boundary_supervision_smoke.yaml")
    baseline_pilot = _load("explicit_offset_boundary_supervision_pilot.yaml")
    smoke = _load("explicit_offset_token_end_geometry_smoke.yaml")
    pilot = _load("explicit_offset_token_end_geometry_pilot.yaml")

    expected_smoke = deepcopy(baseline_smoke)
    expected_smoke["train"]["token_end_geometry_version"] = 1
    expected_smoke["train"]["token_end_premise_path"] = (
        "runs/phase34g-token-end-premise.json"
    )
    expected_smoke["checkpoint"]["dir"] = (
        "checkpoints/explicit_offset_token_end_geometry_smoke"
    )
    expected_pilot = deepcopy(baseline_pilot)
    expected_pilot["train"]["token_end_geometry_version"] = 1
    expected_pilot["train"]["token_end_premise_path"] = (
        "runs/phase34g-token-end-premise.json"
    )
    expected_pilot["checkpoint"]["dir"] = (
        "checkpoints/explicit_offset_token_end_geometry_pilot"
    )

    assert smoke == expected_smoke
    assert pilot == expected_pilot


def test_phase34i_replaces_geometry_with_factorized_heads():
    baseline_smoke = _load("explicit_offset_boundary_supervision_smoke.yaml")
    baseline_pilot = _load("explicit_offset_boundary_supervision_pilot.yaml")
    smoke = _load("explicit_offset_factorized_head_smoke.yaml")
    pilot = _load("explicit_offset_factorized_head_pilot.yaml")

    expected_smoke = deepcopy(baseline_smoke)
    expected_smoke["train"]["factorized_boundary_type_version"] = 1
    expected_smoke["checkpoint"]["dir"] = (
        "checkpoints/explicit_offset_factorized_head_smoke"
    )
    expected_pilot = deepcopy(baseline_pilot)
    expected_pilot["train"]["factorized_boundary_type_version"] = 1
    expected_pilot["checkpoint"]["dir"] = (
        "checkpoints/explicit_offset_factorized_head_pilot"
    )

    assert smoke == expected_smoke
    assert pilot == expected_pilot
