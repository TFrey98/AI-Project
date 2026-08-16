import torch

from story_model.data import (
    ByteBPETokenizer,
    CharTokenizer,
    build_tokenizer,
    get_batch,
    split_text,
    tokenizer_from_dict,
    train_test_split,
)


def test_tokenizer_roundtrip():
    tokenizer = CharTokenizer.from_text("hello world")
    encoded = tokenizer.encode("hello")
    assert tokenizer.decode(encoded) == "hello"


def test_tokenizer_vocab_size():
    tokenizer = CharTokenizer.from_text("aabbcc")
    assert tokenizer.vocab_size == 3


def test_train_test_split():
    data = torch.arange(100)
    train, val = train_test_split(data, 0.9)
    assert len(train) == 90
    assert len(val) == 10


def test_get_batch_shapes():
    data = torch.arange(1000)
    x, y = get_batch(data, block_size=8, batch_size=4)
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(x[0, 1:], y[0, :-1])


def test_split_text_respects_train_split():
    text = "0123456789"
    train_text, val_text = split_text(text, 0.7)

    assert train_text == "0123456"
    assert val_text == "789"


def test_build_tokenizer_defaults_to_char():
    tokenizer = build_tokenizer(
        training_text="aabbcc",
        config=None,
    )

    assert isinstance(tokenizer, CharTokenizer)
    assert tokenizer.vocab_size == 3


def test_build_tokenizer_builds_bpe_from_config():
    tokenizer = build_tokenizer(
        training_text="to be or not to be",
        config={
            "type": "byte_bpe",
            "vocab_size": 260,
            "min_frequency": 2,
        },
    )

    assert isinstance(tokenizer, ByteBPETokenizer)
    assert tokenizer.vocab_size <= 260


def test_tokenizer_from_dict_restores_bpe_tokenizer():
    trained = ByteBPETokenizer.train(
        "to be or not to be",
        vocab_size=260,
    )

    restored = tokenizer_from_dict(trained.to_dict())

    assert isinstance(restored, ByteBPETokenizer)
    assert restored.merges == trained.merges
