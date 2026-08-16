import pytest

from story_model.data import (
    ByteBPETokenizer,
    CharTokenizer,
    merge_token_pair,
)


def test_bpe_roundtrip_ascii():
    text = "to be or not to be"
    tokenizer = ByteBPETokenizer.train(
        text,
        vocab_size=272,
    )

    assert tokenizer.decode(
        tokenizer.encode(text)
    ) == text


def test_bpe_roundtrip_unicode():
    text = "café, storm ⚡, dragon 🐉"
    tokenizer = ByteBPETokenizer.train(
        text,
        vocab_size=280,
        min_frequency=1,
    )

    assert tokenizer.decode(
        tokenizer.encode(text)
    ) == text


def test_bpe_learns_frequent_pair_and_compresses():
    tokenizer = ByteBPETokenizer.train(
        "abababab",
        vocab_size=257,
    )

    assert tokenizer.merges == [
        (ord("a"), ord("b"))
    ]
    assert tokenizer.encode("abababab") == [
        256,
        256,
        256,
        256,
    ]


def test_bpe_training_is_deterministic():
    text = "low lower lowest low lower"

    first = ByteBPETokenizer.train(
        text,
        vocab_size=270,
    )
    second = ByteBPETokenizer.train(
        text,
        vocab_size=270,
    )

    assert first.merges == second.merges


def test_bpe_serialization_roundtrip():
    text = "the theatre and the throne"
    tokenizer = ByteBPETokenizer.train(
        text,
        vocab_size=270,
    )

    restored = ByteBPETokenizer.from_dict(
        tokenizer.to_dict()
    )

    assert restored.merges == tokenizer.merges
    assert restored.encode(text) == tokenizer.encode(text)
    assert restored.decode(restored.encode(text)) == text


def test_bpe_decode_replaces_invalid_utf8():
    tokenizer = ByteBPETokenizer(merges=[])

    assert tokenizer.decode([255]) == "\ufffd"


def test_bpe_rejects_invalid_training_settings():
    with pytest.raises(
        ValueError,
        match="at least 256",
    ):
        ByteBPETokenizer.train("text", vocab_size=255)

    with pytest.raises(
        ValueError,
        match="min_frequency must be positive",
    ):
        ByteBPETokenizer.train(
            "text",
            min_frequency=0,
        )


def test_bpe_rejects_future_token_reference():
    with pytest.raises(
        ValueError,
        match="unavailable token",
    ):
        ByteBPETokenizer(
            merges=[(256, ord("a"))]
        )


def test_pair_merge_is_left_to_right_and_non_overlapping():
    assert merge_token_pair(
        [1, 1, 1],
        pair=(1, 1),
        new_token=2,
    ) == [2, 1]


def test_character_tokenizer_serialization_roundtrip():
    tokenizer = CharTokenizer.from_text("abcabc")
    restored = CharTokenizer.from_dict(
        tokenizer.to_dict()
    )

    assert restored.stoi == tokenizer.stoi
    assert restored.itos == tokenizer.itos
    assert restored.decode(restored.encode("cab")) == "cab"


def test_bpe_special_tokens_are_atomic():
    tokenizer = ByteBPETokenizer(
        merges=[],
        special_tokens=("<|user|>", "<|assistant|>"),
    )

    assert tokenizer.base_vocab_size == 256
    assert tokenizer.vocab_size == 258
    assert tokenizer.special_token_ids == {
        "<|user|>": 256,
        "<|assistant|>": 257,
    }
    assert tokenizer.encode("<|assistant|>") == [257]


def test_bpe_special_token_mixed_text_roundtrip():
    text = "<|user|>CafÃ©?<|assistant|>Oui."
    tokenizer = ByteBPETokenizer(
        merges=[],
        special_tokens=("<|user|>", "<|assistant|>"),
    )

    encoded = tokenizer.encode(text)

    assert 256 in encoded
    assert 257 in encoded
    assert tokenizer.decode(encoded) == text


def test_bpe_prefers_longest_overlapping_special_token():
    tokenizer = ByteBPETokenizer(
        merges=[],
        special_tokens=("<|turn|>", "<|turn|>end"),
    )

    assert tokenizer.encode("<|turn|>end") == [257]


def test_bpe_special_tokens_survive_serialization():
    tokenizer = ByteBPETokenizer(
        merges=[(ord("a"), ord("b"))],
        special_tokens=("<|user|>",),
    )

    restored = ByteBPETokenizer.from_dict(
        tokenizer.to_dict()
    )

    assert restored.merges == tokenizer.merges
    assert restored.special_tokens == tokenizer.special_tokens
    assert restored.special_token_ids == tokenizer.special_token_ids


def test_bpe_rejects_duplicate_special_tokens():
    with pytest.raises(ValueError, match="duplicate special token"):
        ByteBPETokenizer(
            merges=[],
            special_tokens=("<|user|>", "<|user|>"),
        )
