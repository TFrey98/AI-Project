from dataclasses import replace

from story_model.character_data import serialize_character_prompt
from story_model.character_training import load_character_dataset_splits
from story_model.counterfactual_probe import (
    COUNTERFACTUAL_SKILLS,
    build_counterfactual_probe_dataset,
    counterfactual_acceptance_failures,
    counterfactual_probe_splits,
    validate_counterfactual_pairs,
)
from story_model.neutral_instruction import (
    TRAIN_LEXICON,
    VALIDATION_LEXICON,
)
from story_model.neutral_diagnostics import stratified_neutral_pairs


def test_probe_builds_adjacent_minimal_pairs_in_every_split():
    splits = counterfactual_probe_splits(
        train_pairs_per_skill=3,
        validation_pairs_per_skill=2,
        transfer_pairs_per_skill=2,
        seed=17,
    )

    assert len(splits["train"]) == 12
    assert len(splits["val"]) == 8
    assert len(splits["transfer"]) == 8

    for split, records in splits.items():
        report = validate_counterfactual_pairs(records)

        assert report["pairs"] == len(records) // 2
        assert set(report["skills"]) == set(COUNTERFACTUAL_SKILLS)

        for offset in range(0, len(records), 2):
            first, second = records[offset : offset + 2]
            first_prompt = serialize_character_prompt(
                replace(first.context, target_response=None)
            ).splitlines()
            second_prompt = serialize_character_prompt(
                replace(second.context, target_response=None)
            ).splitlines()

            assert sum(
                left != right
                for left, right in zip(first_prompt, second_prompt)
            ) == 1


def test_probe_separates_composition_from_lexical_transfer():
    splits = counterfactual_probe_splits(
        train_pairs_per_skill=20,
        validation_pairs_per_skill=10,
        transfer_pairs_per_skill=10,
        seed=23,
    )
    core_text = " ".join(
        serialize_character_prompt(record.context)
        for split in ("train", "val")
        for record in splits[split]
    )
    transfer_text = " ".join(
        serialize_character_prompt(record.context)
        for record in splits["transfer"]
    )

    assert any(value in core_text for value in TRAIN_LEXICON.organizations)
    assert not any(
        value in core_text for value in VALIDATION_LEXICON.organizations
    )
    assert any(
        value in transfer_text
        for value in VALIDATION_LEXICON.organizations
    )
    assert not any(
        value in transfer_text for value in TRAIN_LEXICON.organizations
    )


def test_probe_diagnostic_sampling_keeps_both_sides_of_each_pair():
    records = counterfactual_probe_splits(
        train_pairs_per_skill=5,
        validation_pairs_per_skill=2,
        transfer_pairs_per_skill=2,
        seed=29,
    )["train"]
    selected = stratified_neutral_pairs(records, pairs_per_skill=2)

    assert len(selected) == 8

    for offset in range(0, len(selected), 2):
        assert (
            selected[offset].conversation_id
            == selected[offset + 1].conversation_id
        )


def test_probe_builder_writes_compatible_manifest_and_transfer_split(
    tmp_path,
):
    manifest = build_counterfactual_probe_dataset(
        tmp_path,
        train_pairs_per_skill=3,
        validation_pairs_per_skill=2,
        transfer_pairs_per_skill=2,
        seed=31,
    )
    training, validation, loaded_manifest = load_character_dataset_splits(
        {
            "type": "character_jsonl",
            "train_path": str(tmp_path / "train.jsonl"),
            "val_path": str(tmp_path / "val.jsonl"),
            "manifest_path": str(tmp_path / "manifest.json"),
        }
    )

    assert loaded_manifest == manifest
    assert len(training) == 12
    assert len(validation) == 8
    assert manifest["train"]["pairs"] == 6
    assert manifest["val"]["pairs"] == 4
    assert manifest["transfer"]["pairs"] == 4
    assert (tmp_path / "transfer.jsonl").is_file()


def test_counterfactual_acceptance_gate_reports_specific_failures():
    passing = {
        split: {
            "exact_response_rate": 1.0,
            "counterfactual_pair_exact_rate": 1.0,
            "end_stop_rate": 1.0,
            "degenerate_loop_rate": 0.0,
            "context_helped_rate": 1.0,
            "mean_context_advantage": 1.0,
        }
        for split in ("train", "val", "transfer")
    }

    assert counterfactual_acceptance_failures(passing) == ()

    failing = {split: dict(values) for split, values in passing.items()}
    failing["val"]["context_helped_rate"] = 0.0
    failures = counterfactual_acceptance_failures(failing)

    assert len(failures) == 1
    assert "val context_helped_rate" in failures[0]
