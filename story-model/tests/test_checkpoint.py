import torch

from story_model.checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from story_model.models.bigram import BigramLanguageModel


def test_checkpoint_contains_version_and_rng(tmp_path):
    model = BigramLanguageModel(vocabulary_size=10)

    path = tmp_path / "versioned.pt"

    save_checkpoint(
        path,
        model,
        step=12,
    )

    checkpoint = read_checkpoint(
        path,
        map_location="cpu",
    )

    assert checkpoint["checkpoint_version"] == 2
    assert checkpoint["step"] == 12
    assert "cpu" in checkpoint["rng_state"]


def test_checkpoint_restores_cpu_rng(tmp_path):
    torch.manual_seed(1234)

    model = BigramLanguageModel(vocabulary_size=10)
    path = tmp_path / "rng.pt"

    save_checkpoint(
        path,
        model,
        step=1,
    )

    expected_random_values = torch.rand(5)

    # Deliberately disturb the random generator.
    torch.manual_seed(9999)
    torch.rand(20)

    restored_model = BigramLanguageModel(
        vocabulary_size=10
    )

    load_checkpoint(
        path,
        restored_model,
        map_location="cpu",
        restore_rng=True,
    )

    actual_random_values = torch.rand(5)

    assert torch.equal(
        expected_random_values,
        actual_random_values,
    )
