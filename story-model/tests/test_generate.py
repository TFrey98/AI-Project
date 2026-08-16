import pytest

from story_model.data import (
    ByteBPETokenizer,
    CharTokenizer,
)
from story_model.generate import (
    encode_prompt,
    load_generation_metadata,
)


def make_tokenizer() -> CharTokenizer:
    return CharTokenizer.from_text("\nABC:")


def test_encode_prompt_accepts_known_characters():
    tokenizer = make_tokenizer()

    assert encode_prompt(
        tokenizer,
        "ABC:\n",
    ) == tokenizer.encode("ABC:\n")


def test_encode_prompt_rejects_empty_text():
    tokenizer = make_tokenizer()

    with pytest.raises(
        ValueError,
        match="prompt cannot be empty",
    ):
        encode_prompt(tokenizer, "")


def test_encode_prompt_rejects_unknown_characters():
    tokenizer = make_tokenizer()

    with pytest.raises(
        ValueError,
        match="outside the checkpoint vocabulary",
    ):
        encode_prompt(tokenizer, "D")


def test_bpe_prompt_accepts_unseen_unicode():
    tokenizer = ByteBPETokenizer(merges=[])
    prompt = "Dragon 🐉"

    encoded = encode_prompt(tokenizer, prompt)

    assert tokenizer.decode(encoded) == prompt


def test_generation_metadata_restores_bpe_tokenizer():
    tokenizer = ByteBPETokenizer.train(
        "to be or not to be",
        vocab_size=260,
    )
    checkpoint = {
        "extra": {
            "config": {"model": {"name": "transformer"}},
            "tokenizer": tokenizer.to_dict(),
        }
    }

    _, restored = load_generation_metadata(
        checkpoint,
        config_path=None,
    )

    assert isinstance(restored, ByteBPETokenizer)
    assert restored.merges == tokenizer.merges