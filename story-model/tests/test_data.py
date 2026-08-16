import pytest
import torch

from story_model.corpus import build_corpus
from story_model.data import (
    ByteBPETokenizer,
    CharTokenizer,
    build_tokenizer,
    get_batch,
    load_text_splits,
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


def test_load_text_splits_supports_explicit_files(tmp_path):
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    train_path.write_text("training", encoding="utf-8")
    val_path.write_text("validation", encoding="utf-8")

    training_text, validation_text = load_text_splits(
        {
            "train_path": train_path,
            "val_path": val_path,
        }
    )

    assert training_text == "training"
    assert validation_text == "validation"


def test_load_text_splits_preserves_legacy_config(tmp_path):
    data_path = tmp_path / "input.txt"
    data_path.write_text("0123456789", encoding="utf-8")

    training_text, validation_text = load_text_splits(
        {
            "path": data_path,
            "train_split": 0.8,
        }
    )

    assert training_text == "01234567"
    assert validation_text == "89"


def test_load_text_splits_rejects_mixed_config(tmp_path):
    data_path = tmp_path / "input.txt"

    with pytest.raises(
        ValueError,
        match="cannot combine",
    ):
        load_text_splits(
            {
                "path": data_path,
                "train_path": data_path,
                "val_path": data_path,
                "train_split": 0.9,
            }
        )


def test_load_text_splits_requires_both_explicit_paths(tmp_path):
    with pytest.raises(
        ValueError,
        match="provide both",
    ):
        load_text_splits(
            {"train_path": tmp_path / "train.txt"}
        )


def test_split_text_rejects_invalid_fraction():
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        split_text("text", 1.0)


def test_load_text_splits_verifies_corpus_manifest(tmp_path):
    raw_path = tmp_path / "raw"
    output_path = tmp_path / "processed"
    raw_path.mkdir()
    (raw_path / "first.txt").write_text(
        "First document.\n" * 10,
        encoding="utf-8",
    )
    (raw_path / "second.txt").write_text(
        "Second document.\n" * 10,
        encoding="utf-8",
    )
    build_corpus(raw_path, output_path)

    config = {
        "train_path": output_path / "train.txt",
        "val_path": output_path / "val.txt",
        "manifest_path": output_path / "manifest.json",
    }

    load_text_splits(config)
    (output_path / "train.txt").write_text(
        "changed corpus",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="does not match its manifest",
    ):
        load_text_splits(config)
