import json
from pathlib import Path

from story_model.character_training import load_character_dataset_splits
from story_model.lexical_copy import (
    COPY_COLOR_VALUES,
    COPY_ROUTE_VALUES,
    build_lexical_copy_dataset,
    lexical_copy_curriculum,
)
from story_model.semantic_evaluation import (
    score_diagnostic_rows,
    summarize_semantic_rows,
)
from story_model.semantic_transfer import semantic_transfer_splits


def test_copy_pools_are_large_compositional_and_distinct():
    assert len(COPY_COLOR_VALUES) == 384
    assert len(COPY_ROUTE_VALUES) == 384
    assert len(set(COPY_COLOR_VALUES)) == len(COPY_COLOR_VALUES)
    assert len(set(COPY_ROUTE_VALUES)) == len(COPY_ROUTE_VALUES)
    assert all(" " in value for value in COPY_COLOR_VALUES)
    assert all(" " in value for value in COPY_ROUTE_VALUES)


def test_phase29_keeps_all_phase28_diagnostic_rows_unchanged():
    baseline = semantic_transfer_splits(
        train_pairs_per_skill=20,
        validation_pairs_per_skill=5,
        lexical_pairs_per_skill=5,
        paraphrase_pairs_per_skill=5,
        transfer_pairs_per_skill=5,
        seed=17,
    )
    phase29, _ = lexical_copy_curriculum(
        copy_pairs_per_skill=30,
        anchor_pairs_per_skill=10,
        baseline_train_pairs_per_skill=20,
        validation_pairs_per_skill=5,
        lexical_pairs_per_skill=5,
        paraphrase_pairs_per_skill=5,
        transfer_pairs_per_skill=5,
        seed=17,
    )

    for split in ("val", "lexical", "paraphrase", "transfer"):
        assert phase29[split] == baseline[split]


def test_sparse_training_keys_cover_rows_and_reference_values():
    splits, answer_keys = lexical_copy_curriculum(
        copy_pairs_per_skill=300,
        anchor_pairs_per_skill=100,
        baseline_train_pairs_per_skill=20,
        validation_pairs_per_skill=5,
        lexical_pairs_per_skill=5,
        paraphrase_pairs_per_skill=5,
        transfer_pairs_per_skill=5,
        seed=23,
    )
    entries = answer_keys["entries"]
    expected_count = sum(len(records) for records in splits.values())

    assert len(entries) == expected_count
    copy_values = set(COPY_COLOR_VALUES + COPY_ROUTE_VALUES)
    exposed = {
        entry["expected_value"]
        for entry in entries.values()
        if entry["split"] == "train"
        and entry["expected_value"] in copy_values
    }
    assert len(exposed) >= 500

    rows = []

    for record in splits["train"]:
        key = entries[record.context.context_id]
        rows.append(
            {
                "scenario_id": record.context.context_id,
                "split": "train",
                "conversation_id": record.conversation_id,
                "skill": key["skill"],
                "full_context": {
                    "response": record.context.target_response,
                },
            }
        )

    scored = score_diagnostic_rows(rows, entries)
    summary = summarize_semantic_rows(scored)["train"]

    assert summary["semantic_correct_rate"] == 1.0
    assert summary["counterfactual_pair_semantic_rate"] == 1.0


def test_lexical_copy_builder_writes_compatible_manifest(tmp_path: Path):
    manifest = build_lexical_copy_dataset(
        tmp_path,
        copy_pairs_per_skill=30,
        anchor_pairs_per_skill=10,
        baseline_train_pairs_per_skill=20,
        validation_pairs_per_skill=5,
        lexical_pairs_per_skill=5,
        paraphrase_pairs_per_skill=5,
        transfer_pairs_per_skill=5,
        seed=29,
    )
    training, validation, loaded_manifest = load_character_dataset_splits(
        {
            "type": "character_jsonl",
            "train_path": str(tmp_path / "train.jsonl"),
            "val_path": str(tmp_path / "val.jsonl"),
            "manifest_path": str(tmp_path / "manifest.json"),
        }
    )
    answer_keys = json.loads(
        (tmp_path / "answer_keys.json").read_text(encoding="utf-8")
    )

    assert loaded_manifest == manifest
    assert len(training) == 160
    assert len(validation) == 20
    assert len(answer_keys["entries"]) == 240
    assert manifest["training_pairs_per_skill"] == {
        "copy": 30,
        "anchor": 10,
    }
