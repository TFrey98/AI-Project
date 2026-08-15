"""Verify that a small Transformer can memorize one fixed batch."""

import torch

from story_model.models.transformer import TransformerLanguageModel
from story_model.runtime import resolve_device


def main() -> None:
    torch.manual_seed(1337)

    device = resolve_device("auto")

    model = TransformerLanguageModel(
        vocabulary_size=10,
        block_size=8,
        embedding_dim=32,
        attention_heads=4,
        layers=2,
        feed_forward_dim=64,
        dropout=0.0,
    ).to(device)

    # A fixed numerical sequence and its next-token targets.
    inputs = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7]],
        dtype=torch.long,
        device=device,
    ).repeat(4, 1)

    targets = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8]],
        dtype=torch.long,
        device=device,
    ).repeat(4, 1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-3,
        weight_decay=0.0,
    )

    initial_loss = None

    model.train()

    for step in range(301):
        _, loss = model(inputs, targets)

        assert loss is not None

        if initial_loss is None:
            initial_loss = loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(
                f"step {step:3d}: "
                f"loss {loss.item():.6f}"
            )

    model.eval()

    with torch.no_grad():
        _, final_loss_tensor = model(inputs, targets)

    assert final_loss_tensor is not None
    final_loss = final_loss_tensor.item()

    print(f"device: {device}")
    print(f"initial loss: {initial_loss:.6f}")
    print(f"final loss: {final_loss:.6f}")

    if final_loss >= 0.1:
        raise RuntimeError(
            "The Transformer failed to overfit the fixed batch"
        )

    print("single-batch overfit: passed")


if __name__ == "__main__":
    main()