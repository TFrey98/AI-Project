import torch

from story_model.models.attention import AttentionLanguageModel


def test_attention_forward_shapes():
    model = AttentionLanguageModel(
        vocabulary_size=10,
        block_size=8,
        embedding_dim=16,
    )

    tokens = torch.randint(0, 10, (2, 8))
    targets = torch.randint(0, 10, (2, 8))

    logits, loss = model(tokens, targets)

    assert logits.shape == (2, 8, 10)
    assert loss is not None
    assert torch.isfinite(loss)


def test_attention_cannot_see_future_tokens():
    torch.manual_seed(7)

    model = AttentionLanguageModel(
        vocabulary_size=10,
        block_size=5,
        embedding_dim=16,
        dropout=0.0,
    )
    model.eval()

    first = torch.tensor([[1, 2, 3, 4, 5]])
    second = torch.tensor([[1, 2, 3, 8, 9]])

    first_logits, _ = model(first)
    second_logits, _ = model(second)

    assert torch.allclose(
        first_logits[:, :3],
        second_logits[:, :3],
        atol=1e-6,
    )


def test_attention_generate_extends_sequence():
    model = AttentionLanguageModel(
        vocabulary_size=10,
        block_size=8,
        embedding_dim=16,
    )

    tokens = torch.zeros((1, 1), dtype=torch.long)
    output = model.generate(tokens, max_new_tokens=5)

    assert output.shape == (1, 6)