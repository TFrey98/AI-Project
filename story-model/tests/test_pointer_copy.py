from dataclasses import replace

import torch

from story_model.character_data import (
    CHARACTER_CONTROL_TOKENS,
    serialize_character_prompt,
)
from story_model.character_training import (
    character_training_record_from_json,
    character_training_record_to_json,
    encode_character_training_record,
)
from story_model.checkpoint import load_model_warm_start
from story_model.data import ByteBPETokenizer
from story_model.models import build_model
from story_model.pointer_copy import pointer_copy_splits
from story_model.semantic_transfer import semantic_transfer_splits


def pointer_model(vocabulary_size: int = 64, block_size: int = 16):
    return build_model(
        {
            "name": "transformer",
            "embedding_dim": 16,
            "attention_heads": 4,
            "layers": 1,
            "feed_forward_dim": 32,
            "dropout": 0.0,
            "position_encoding": "rope",
            "copy_mechanism": "pointer_generator",
            "copy_loss_weight": 0.25,
        },
        vocabulary_size=vocabulary_size,
        block_size=block_size,
    )


def small_splits(seed: int = 17):
    return pointer_copy_splits(
        train_pairs_per_skill=1,
        validation_pairs_per_skill=1,
        lexical_pairs_per_skill=1,
        paraphrase_pairs_per_skill=1,
        transfer_pairs_per_skill=1,
        seed=seed,
    )


def test_phase30_only_adds_copy_annotations_to_phase28_rows():
    baseline = semantic_transfer_splits(
        train_pairs_per_skill=1,
        validation_pairs_per_skill=1,
        lexical_pairs_per_skill=1,
        paraphrase_pairs_per_skill=1,
        transfer_pairs_per_skill=1,
        seed=17,
    )
    annotated, keys = small_splits(17)

    for split, records in annotated.items():
        for source, record in zip(baseline[split], records):
            assert replace(record, copy_value=None) == source
            assert record.copy_value == keys["entries"][
                record.context.context_id
            ]["expected_value"]


def test_copy_value_json_is_optional_and_roundtrips():
    record = small_splits()[0]["train"][0]
    encoded = character_training_record_to_json(record)
    assert character_training_record_from_json(encoded) == record
    assert '"copy_value"' in encoded
    assert '"copy_value"' not in character_training_record_to_json(
        replace(record, copy_value=None)
    )


def test_copy_encoding_survives_incompatible_bpe_boundaries():
    records, _ = small_splits(1337)
    record = next(
        item
        for item in records["train"]
        if "supplied_fact" in item.context.context_id
    )
    assert record.copy_value == "lime"
    tokenizer = ByteBPETokenizer.train(
        ("uses lime to " * 1000) + ("with lime. " * 100),
        vocab_size=280,
    ).with_special_tokens(CHARACTER_CONTROL_TOKENS)
    example = encode_character_training_record(record, tokenizer, 1024)
    copied = [
        index
        for index, source in enumerate(example.copy_targets)
        if source != -100
    ]
    assert example.copy_supervised_tokens == len(b"lime")
    assert [example.target_ids[index] for index in copied] == list(b"lime")

    for target_index in copied:
        source = example.copy_targets[target_index]
        token_position = source // example.copy_source_width
        byte_offset = source % example.copy_source_width
        source_bytes = tokenizer.token_bytes(
            example.input_ids[token_position]
        )
        assert source_bytes[byte_offset] == example.target_ids[target_index]


def test_copy_encoding_maps_to_prompt_positions():
    record = small_splits(29)[0]["train"][0]
    material = (
        serialize_character_prompt(record.context)
        + (record.context.target_response or "")
    ) * 8
    tokenizer = ByteBPETokenizer.train(
        material, vocab_size=288
    ).with_special_tokens(CHARACTER_CONTROL_TOKENS)
    example = encode_character_training_record(record, tokenizer, 1024)
    assert example.copy_supervised_tokens > 0

    for source in example.copy_targets:
        if source != -100:
            assert example.copy_source_mask[
                source // example.copy_source_width
            ]


def test_pointer_distribution_can_emit_prompt_byte():
    model = pointer_model(vocabulary_size=16, block_size=4)
    model.eval()

    with torch.no_grad():
        model.copy_query.weight.zero_()
        model.copy_key.weight.zero_()
        model.copy_gate.weight.zero_()
        model.copy_gate.bias.fill_(-20.0)

    tokens = torch.tensor([[7, 2, 3, 4]])
    source_mask = torch.tensor([[True, False, False, False]])
    logits, _ = model(tokens, copy_source_mask=source_mask)
    assert int(logits[0, -1].argmax()) == 7


def test_pointer_auxiliary_reaches_copy_parameters():
    model = pointer_model(vocabulary_size=16, block_size=4)
    tokens = torch.tensor([[7, 2, 3, 4]])
    targets = torch.tensor([[-100, -100, 7, 5]])
    source_mask = torch.tensor([[True, True, False, False]])
    copy_targets = torch.tensor([[-100, -100, 0, -100]])
    _, loss = model(
        tokens,
        targets,
        copy_source_mask=source_mask,
        copy_targets=copy_targets,
    )
    assert loss is not None and torch.isfinite(loss)
    loss.backward()

    for parameter in (
        model.copy_query.weight,
        model.copy_key.weight,
        model.copy_offset_embedding.weight,
        model.copy_gate.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_foundation_warm_start_keeps_pointer_parameters():
    base_config = {
        "name": "transformer",
        "embedding_dim": 16,
        "attention_heads": 4,
        "layers": 1,
        "feed_forward_dim": 32,
        "dropout": 0.0,
        "position_encoding": "rope",
    }
    source = build_model(base_config, vocabulary_size=10, block_size=8)
    destination = pointer_model(vocabulary_size=12, block_size=8)
    initial = destination.copy_query.weight.detach().clone()
    expanded = load_model_warm_start(
        destination,
        {"model_state_dict": source.state_dict()},
        source_vocabulary_size=10,
        destination_vocabulary_size=12,
        allowed_new_parameter_prefixes=(
            "copy_query.",
            "copy_key.",
            "copy_offset_embedding.",
            "copy_gate.",
        ),
    )
    assert torch.equal(destination.copy_query.weight, initial)
    assert set(expanded) == {
        "token_embeddings.weight",
        "output_projection.weight",
        "output_projection.bias",
    }
