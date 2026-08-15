import torch

from story_model.models.bigram import BigramLanguageModel


def test_forward_without_targets():
    model = BigramLanguageModel(vocabulary_size=10)
    idx = torch.randint(0, 10, (2, 5))
    logits, loss = model(idx)
    assert logits.shape == (2, 5, 10)
    assert loss is None


def test_forward_with_targets():
    model = BigramLanguageModel(vocabulary_size=10)
    idx = torch.randint(0, 10, (2, 5))
    targets = torch.randint(0, 10, (2, 5))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 5, 10)
    assert loss is not None
    assert loss.item() > 0


def test_generate_extends_sequence():
    model = BigramLanguageModel(vocabulary_size=10)
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 6)
