import torch

from story_model.checkpoint import load_checkpoint, save_checkpoint
from story_model.models.bigram import BigramLanguageModel


def test_save_and_load_roundtrip(tmp_path):
    model = BigramLanguageModel(vocabulary_size=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = tmp_path / "checkpoints" / "test.pt"
    save_checkpoint(ckpt_path, model, optimizer, step=42, extra={"note": "test"})
    assert ckpt_path.exists()

    loaded_model = BigramLanguageModel(vocabulary_size=10)
    loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=1e-3)
    checkpoint = load_checkpoint(ckpt_path, loaded_model, loaded_optimizer)

    assert checkpoint["step"] == 42
    assert checkpoint["extra"] == {"note": "test"}
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2)
