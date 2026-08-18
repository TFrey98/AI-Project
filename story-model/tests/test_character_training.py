from dataclasses import replace

import pytest

from story_model.character_data import (
    ASSISTANT_MARKER,
    CHARACTER_CONTROL_TOKENS,
    END_MARKER,
    CharacterContext,
    CharacterProfile,
    ConversationTurn,
    RelationshipState,
    SceneState,
)
from story_model.character_training import (
    CHARACTER_BEHAVIOR_TAGS,
    RESPONSE_IGNORE_INDEX,
    CharacterTrainingRecord,
    audit_character_training_records,
    build_character_dataset,
    character_training_record_from_json,
    character_training_record_to_json,
    encode_character_training_record,
    get_character_batch,
    load_character_dataset_splits,
    load_character_training_records,
    save_character_training_records,
    split_character_training_records,
    summarize_character_examples,
)
from story_model.data import ByteBPETokenizer


def build_context(
    context_id: str = "conv_a_001",
) -> CharacterContext:
    return CharacterContext(
        context_id=context_id,
        character=CharacterProfile(
            character_id="elara",
            name="Elara",
            summary="A guarded investigator.",
            traits=("guarded",),
            voice=("formal",),
        ),
        relationship=RelationshipState(
            character_id="elara",
            participant_id="traveler",
            participant_name="Traveler",
            attitude="Cautiously cooperative.",
        ),
        scene=SceneState(
            location="Castle gate",
            time="Night",
            participants=("elara", "traveler"),
        ),
        recent_turns=(
            ConversationTurn(
                role="user",
                speaker_id="traveler",
                text="The gate is barred.",
            ),
            ConversationTurn(
                role="assistant",
                speaker_id="elara",
                text="Then we find another entrance.",
            ),
            ConversationTurn(
                role="user",
                speaker_id="traveler",
                text="Do you know one?",
            ),
        ),
        target_response="I know one, but first answer my question.",
    )


def build_tokenizer() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(
        "The gate is barred and another entrance is hidden. " * 4,
        vocab_size=272,
    ).with_special_tokens(CHARACTER_CONTROL_TOKENS)


def test_character_training_record_json_roundtrip():
    record = CharacterTrainingRecord("conv_a", build_context())
    encoded = character_training_record_to_json(record)

    assert character_training_record_from_json(encoded) == record
    assert character_training_record_to_json(
        character_training_record_from_json(encoded)
    ) == encoded


def test_character_training_record_requires_target():
    context = replace(build_context(), target_response=None)

    with pytest.raises(ValueError, match="target_response"):
        CharacterTrainingRecord("conv_a", context)


def test_character_training_record_validates_behavior_tags():
    record = CharacterTrainingRecord(
        "conv_a",
        build_context(),
        behavior_tags=("voice", "memory"),
    )
    restored = character_training_record_from_json(
        character_training_record_to_json(record)
    )

    assert restored.behavior_tags == ("voice", "memory")

    with pytest.raises(ValueError, match="unknown character behavior"):
        CharacterTrainingRecord(
            "conv_a",
            build_context("unknown_tag"),
            behavior_tags=("improvisation",),
        )


def test_character_jsonl_file_roundtrip(tmp_path):
    records = (
        CharacterTrainingRecord("conv_a", build_context()),
        CharacterTrainingRecord(
            "conv_b",
            build_context("conv_b_001"),
        ),
    )
    path = tmp_path / "records.jsonl"

    save_character_training_records(records, path)

    assert load_character_training_records(path) == records


def test_character_jsonl_rejects_duplicate_context_ids(tmp_path):
    record = CharacterTrainingRecord("conv_a", build_context())
    path = tmp_path / "duplicates.jsonl"
    path.write_text(
        character_training_record_to_json(record) * 2,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate context_id"):
        load_character_training_records(path)


def test_conversation_split_has_no_leakage_and_is_deterministic():
    records = tuple(
        CharacterTrainingRecord(
            f"conv_{conversation}",
            build_context(f"conv_{conversation}_{turn}"),
        )
        for conversation in range(4)
        for turn in range(2)
    )

    first = split_character_training_records(
        records,
        validation_fraction=0.25,
        seed=7,
    )
    second = split_character_training_records(
        records,
        validation_fraction=0.25,
        seed=7,
    )
    train_ids = {record.conversation_id for record in first[0]}
    val_ids = {record.conversation_id for record in first[1]}

    assert first == second
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {
        f"conv_{index}" for index in range(4)
    }


def test_conversation_split_requires_distinct_conversations():
    records = (
        CharacterTrainingRecord("same", build_context()),
        CharacterTrainingRecord(
            "same",
            build_context("conv_a_002"),
        ),
    )

    with pytest.raises(ValueError, match="two conversations"):
        split_character_training_records(records)


def test_character_dataset_builds_manifest_and_disjoint_files(tmp_path):
    records = tuple(
        CharacterTrainingRecord(
            f"conv_{index}",
            build_context(f"context_{index}"),
        )
        for index in range(3)
    )

    manifest = build_character_dataset(records, tmp_path, seed=9)
    training = load_character_training_records(
        tmp_path / "train.jsonl"
    )
    validation = load_character_training_records(
        tmp_path / "val.jsonl"
    )

    assert manifest["source_examples"] == 3
    assert (tmp_path / "manifest.json").exists()
    assert {
        record.conversation_id for record in training
    }.isdisjoint(
        {record.conversation_id for record in validation}
    )


def test_character_dataset_loader_verifies_manifest(tmp_path):
    records = tuple(
        CharacterTrainingRecord(
            f"conv_{index}",
            build_context(f"context_{index}"),
        )
        for index in range(3)
    )
    build_character_dataset(records, tmp_path, seed=9)

    training, validation, manifest = load_character_dataset_splits(
        {
            "type": "character_jsonl",
            "train_path": tmp_path / "train.jsonl",
            "val_path": tmp_path / "val.jsonl",
            "manifest_path": tmp_path / "manifest.json",
        }
    )

    assert manifest is not None
    assert len(training) + len(validation) == 3


def test_character_dataset_loader_rejects_manifest_tampering(tmp_path):
    records = tuple(
        CharacterTrainingRecord(
            f"conv_{index}",
            build_context(f"context_{index}"),
        )
        for index in range(3)
    )
    build_character_dataset(records, tmp_path, seed=9)
    train_path = tmp_path / "train.jsonl"
    training = load_character_training_records(train_path)
    changed = replace(
        training[0],
        context=replace(
            training[0].context,
            target_response="A changed response.",
        ),
    )
    save_character_training_records(
        (changed, *training[1:]),
        train_path,
    )

    with pytest.raises(ValueError, match="does not match its manifest"):
        load_character_dataset_splits(
            {
                "type": "character_jsonl",
                "train_path": train_path,
                "val_path": tmp_path / "val.jsonl",
                "manifest_path": tmp_path / "manifest.json",
            }
        )


def test_character_dataset_loader_rejects_conversation_leakage(tmp_path):
    train_record = CharacterTrainingRecord(
        "shared",
        build_context("train_context"),
    )
    val_record = CharacterTrainingRecord(
        "shared",
        build_context("val_context"),
    )
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    save_character_training_records((train_record,), train_path)
    save_character_training_records((val_record,), val_path)

    with pytest.raises(ValueError, match="conversation leakage"):
        load_character_dataset_splits(
            {
                "type": "character_jsonl",
                "train_path": train_path,
                "val_path": val_path,
            }
        )


def test_response_encoding_masks_prompt_and_padding():
    tokenizer = build_tokenizer()
    record = CharacterTrainingRecord("conv_a", build_context())
    example = encode_character_training_record(
        record,
        tokenizer,
        block_size=512,
    )
    assistant_id = tokenizer.special_token_ids[ASSISTANT_MARKER]
    cue_index = max(
        index
        for index, token_id in enumerate(example.input_ids)
        if token_id == assistant_id
    )
    supervised = [
        token_id
        for token_id in example.target_ids
        if token_id != RESPONSE_IGNORE_INDEX
    ]

    assert len(example.input_ids) == 512
    assert len(example.target_ids) == 512
    assert all(
        target == RESPONSE_IGNORE_INDEX
        for target in example.target_ids[:cue_index]
    )
    assert example.target_ids[cue_index] != RESPONSE_IGNORE_INDEX
    assert all(
        target == RESPONSE_IGNORE_INDEX
        for target in example.target_ids[example.sequence_tokens :]
    )
    assert example.supervised_tokens == len(supervised)
    assert tokenizer.special_token_ids[END_MARKER] in supervised


def test_response_encoding_drops_only_complete_oldest_turns():
    tokenizer = build_tokenizer()
    context = build_context()
    short_context = replace(
        context,
        recent_turns=context.recent_turns[-1:],
    )
    short_record = CharacterTrainingRecord("conv_a", short_context)
    short_example = encode_character_training_record(
        short_record,
        tokenizer,
        block_size=512,
    )
    record = CharacterTrainingRecord("conv_a", context)
    example = encode_character_training_record(
        record,
        tokenizer,
        block_size=short_example.sequence_tokens,
    )
    visible_text = tokenizer.decode(
        list(example.input_ids[: example.sequence_tokens])
    )

    assert example.dropped_turns == 2
    assert "Do you know one?" in visible_text
    assert "The gate is barred." not in visible_text


def test_character_batch_and_summary_shapes():
    tokenizer = build_tokenizer()
    records = (
        CharacterTrainingRecord("conv_a", build_context()),
        CharacterTrainingRecord(
            "conv_b",
            build_context("conv_b_001"),
        ),
    )
    examples = tuple(
        encode_character_training_record(
            record,
            tokenizer,
            block_size=512,
        )
        for record in records
    )

    inputs, targets = get_character_batch(examples, batch_size=3)
    summary = summarize_character_examples(records, examples)

    assert inputs.shape == (3, 512)
    assert targets.shape == (3, 512)
    assert summary["examples"] == 2
    assert summary["conversations"] == 2
    assert summary["supervised_tokens"] > 0


def test_character_dataset_audit_passes_covered_fixture():
    tokenizer = build_tokenizer()
    records = (
        CharacterTrainingRecord(
            "conv_a",
            build_context(),
            behavior_tags=("voice",),
        ),
        CharacterTrainingRecord(
            "conv_b",
            build_context("conv_b_001"),
            behavior_tags=("memory",),
        ),
    )
    report = audit_character_training_records(
        records,
        tokenizer,
        block_size=512,
        min_examples=2,
        min_conversations=2,
        min_examples_per_tag=1,
        required_tags=("voice", "memory"),
    )

    assert report["passed"]
    assert not report["errors"]
    assert report["tag_counts"]["voice"] == 1
    assert report["tag_counts"]["memory"] == 1


def test_character_dataset_audit_rejects_missing_coverage():
    tokenizer = build_tokenizer()
    record = CharacterTrainingRecord("conv_a", build_context())
    report = audit_character_training_records(
        (record,),
        tokenizer,
        block_size=512,
        min_examples=2,
        min_conversations=2,
        min_examples_per_tag=1,
        required_tags=CHARACTER_BEHAVIOR_TAGS,
    )

    assert not report["passed"]
    assert report["errors"]
    assert report["untagged_examples"] == 1
