from collections import Counter

from story_model.character_training import (
    CHARACTER_BEHAVIOR_TAGS,
    load_character_dataset_splits,
)
from story_model.neutral_instruction import (
    NEUTRAL_SKILLS,
    build_neutral_instruction_dataset,
    neutral_instruction_records,
)


def test_neutral_curriculum_has_expected_skill_coverage():
    records = neutral_instruction_records(
        "train",
        examples_per_skill=3,
    )
    skills = Counter(
        "_".join(record.context.context_id.split("_")[2:-1])
        for record in records
    )
    tags = {
        tag for record in records for tag in record.behavior_tags
    }

    assert len(records) == 3 * len(NEUTRAL_SKILLS)
    assert skills == {skill: 3 for skill in NEUTRAL_SKILLS}
    assert tags == CHARACTER_BEHAVIOR_TAGS


def test_neutral_validation_holds_out_words_and_question_templates():
    training = neutral_instruction_records("train", 4)
    validation = neutral_instruction_records("val", 4)
    training_questions = {
        record.context.recent_turns[-1].text for record in training
    }
    validation_questions = {
        record.context.recent_turns[-1].text for record in validation
    }

    assert training_questions.isdisjoint(validation_questions)
    assert {
        record.context.context_id for record in training
    }.isdisjoint(
        {record.context.context_id for record in validation}
    )


def test_privacy_examples_do_not_reveal_the_protected_name():
    records = neutral_instruction_records("val", 5)
    privacy_records = tuple(
        record
        for record in records
        if "boundary" in record.behavior_tags
    )

    assert len(privacy_records) == 5

    for record in privacy_records:
        secret = record.context.memories[0].content
        protected_name = secret.split()[0]

        assert protected_name not in record.context.target_response
        assert "protected" in record.context.target_response or (
            "cannot disclose" in record.context.target_response
        )


def test_builder_writes_explicit_leakage_safe_splits(tmp_path):
    manifest = build_neutral_instruction_dataset(
        tmp_path,
        train_examples_per_skill=3,
        validation_examples_per_skill=2,
    )
    training, validation, loaded_manifest = (
        load_character_dataset_splits(
            {
                "type": "character_jsonl",
                "train_path": str(tmp_path / "train.jsonl"),
                "val_path": str(tmp_path / "val.jsonl"),
                "manifest_path": str(tmp_path / "manifest.json"),
            }
        )
    )

    assert loaded_manifest == manifest
    assert manifest["split_strategy"] == (
        "held_out_lexicon_and_templates"
    )
    assert manifest["train"]["examples"] == 30
    assert manifest["val"]["examples"] == 20
    assert set(manifest["train"]["skills"]) == set(NEUTRAL_SKILLS)
    assert {
        record.conversation_id for record in training
    }.isdisjoint(
        {record.conversation_id for record in validation}
    )
