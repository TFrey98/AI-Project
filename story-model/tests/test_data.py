import torch

from story_model.data import CharTokenizer, get_batch, train_test_split


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
