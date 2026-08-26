from pathlib import Path

from story_model.character_training import load_character_dataset_splits
from story_model.neutral_instruction import VALIDATION_LEXICON
from story_model.neutral_diagnostics import neutral_skill
from story_model.semantic_transfer import (
    DEVELOPMENT_LEXICON,
    SEMANTIC_TRANSFER_SPLITS,
    build_semantic_answer_keys,
    build_semantic_transfer_dataset,
    semantic_transfer_splits,
)


def _text(records) -> str:
    return " ".join(
        (
            record.context.recent_turns[-1].text
            + " "
            + (record.context.target_response or "")
        )
        for record in records
    )


def test_semantic_transfer_factorizes_lexical_and_paraphrase_axes():
    splits = semantic_transfer_splits(
        train_pairs_per_skill=5,
        validation_pairs_per_skill=3,
        lexical_pairs_per_skill=3,
        paraphrase_pairs_per_skill=3,
        transfer_pairs_per_skill=3,
        seed=17,
    )

    assert tuple(splits) == SEMANTIC_TRANSFER_SPLITS
    assert len(splits["train"]) == 20
    assert all(len(splits[split]) == 12 for split in splits if split != "train")
    familiar_text = _text(splits["train"] + splits["val"])
    lexical_text = _text(splits["lexical"])
    paraphrase_text = _text(splits["paraphrase"])
    transfer_text = _text(splits["transfer"])

    assert any(value in familiar_text for value in DEVELOPMENT_LEXICON.colors)
    assert not any(value in familiar_text for value in VALIDATION_LEXICON.colors)
    assert any(value in lexical_text for value in VALIDATION_LEXICON.colors)
    assert any(value in paraphrase_text for value in DEVELOPMENT_LEXICON.colors)
    assert not any(value in paraphrase_text for value in VALIDATION_LEXICON.colors)
    assert any(value in transfer_text for value in VALIDATION_LEXICON.colors)

    for split in SEMANTIC_TRANSFER_SPLITS:
        assert neutral_skill(splits[split][0]) in {
            "scene_route",
            "supplied_fact",
        }


def test_semantic_answer_keys_cover_every_row_and_pair_foil():
    splits = semantic_transfer_splits(
        train_pairs_per_skill=3,
        validation_pairs_per_skill=2,
        lexical_pairs_per_skill=2,
        paraphrase_pairs_per_skill=2,
        transfer_pairs_per_skill=2,
        seed=23,
    )
    payload = build_semantic_answer_keys(splits)
    entries = payload["entries"]

    assert len(entries) == sum(len(records) for records in splits.values())

    for records in splits.values():
        for offset in range(0, len(records), 2):
            first, second = records[offset : offset + 2]
            first_key = entries[first.context.context_id]
            second_key = entries[second.context.context_id]

            assert first_key["expected_value"] == second_key["alternative_value"]
            assert second_key["expected_value"] == first_key["alternative_value"]
            assert first_key["expected_value"] != second_key["expected_value"]


def test_semantic_transfer_builder_writes_training_manifest_and_all_axes(
    tmp_path: Path,
):
    manifest = build_semantic_transfer_dataset(
        tmp_path,
        train_pairs_per_skill=3,
        validation_pairs_per_skill=2,
        lexical_pairs_per_skill=2,
        paraphrase_pairs_per_skill=2,
        transfer_pairs_per_skill=2,
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

    assert loaded_manifest == manifest
    assert len(training) == 12
    assert len(validation) == 8
    assert manifest["source_examples"] == 20
    assert manifest["diagnostic_examples"] == 24
    assert (tmp_path / "answer_keys.json").is_file()

    for split in SEMANTIC_TRANSFER_SPLITS:
        assert (tmp_path / f"{split}.jsonl").is_file()
