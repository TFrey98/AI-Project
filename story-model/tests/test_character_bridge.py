from collections import Counter

from story_model.character_bridge import bridge_records
from story_model.character_training import (
    CHARACTER_BEHAVIOR_TAGS,
    build_character_dataset,
    load_character_training_records,
)


def test_bridge_records_cover_every_behavior_in_every_conversation():
    records = bridge_records()
    conversations = {
        record.conversation_id for record in records
    }
    characters = {
        record.context.character.character_id for record in records
    }
    counts = Counter(
        tag for record in records for tag in record.behavior_tags
    )

    assert len(records) == 120
    assert len(conversations) == 24
    assert len(characters) == 4
    assert counts == {
        tag: 24 for tag in CHARACTER_BEHAVIOR_TAGS
    }
    assert all(
        sum(
            record.conversation_id == conversation_id
            for record in records
        )
        == 5
        for conversation_id in conversations
    )


def test_bridge_long_context_targets_retain_the_first_question():
    long_records = tuple(
        record
        for record in bridge_records()
        if "long_context" in record.behavior_tags
    )

    assert len(long_records) == 24

    for record in long_records:
        context = record.context

        assert len(context.recent_turns) == 9
        assert context.recent_turns[0].role == "user"
        assert context.recent_turns[0].text in context.target_response
        assert context.character.goals[0][:-1].lower() in (
            context.target_response.lower()
        )


def test_bridge_split_is_leakage_safe_and_balanced(tmp_path):
    manifest = build_character_dataset(
        bridge_records(),
        output_dir=tmp_path,
        validation_fraction=0.2,
        seed=1337,
    )
    training = load_character_training_records(
        tmp_path / "train.jsonl"
    )
    validation = load_character_training_records(
        tmp_path / "val.jsonl"
    )
    training_conversations = {
        record.conversation_id for record in training
    }
    validation_conversations = {
        record.conversation_id for record in validation
    }

    assert manifest["train"]["examples"] == 95
    assert manifest["val"]["examples"] == 25
    assert len(training_conversations) == 19
    assert len(validation_conversations) == 5
    assert training_conversations.isdisjoint(validation_conversations)
    assert manifest["train"]["behavior_tags"] == {
        tag: 19 for tag in CHARACTER_BEHAVIOR_TAGS
    }
    assert manifest["val"]["behavior_tags"] == {
        tag: 5 for tag in CHARACTER_BEHAVIOR_TAGS
    }
    assert {
        record.context.character.character_id
        for record in validation
    } == {"bram", "corvin", "elara", "mirelle"}
