"""Prove pointer copying on held-out combinations of familiar pieces."""

from __future__ import annotations

import torch

from story_model.models import build_model
from story_model.runtime import resolve_device, seed_everything


FIRST = tuple(range(10, 18))
SECOND = tuple(range(20, 28))
END = 6


def example(left: int, right: int):
    sequence = [1, left, right, 2, left, right, END]
    inputs = sequence[:-1]
    targets = sequence[1:]
    targets[:3] = [-100] * 3
    source_mask = [True, True, True, False, False, False]
    copy_targets = [-100, -100, -100, 1, 2, -100]
    return inputs, targets, source_mask, copy_targets


@torch.no_grad()
def generate(model, left: int, right: int, device: str) -> tuple[int, ...]:
    tokens = torch.tensor([[1, left, right, 2]], device=device)
    source_mask = torch.tensor(
        [[True, True, True, False]],
        device=device,
    )
    generated = []

    for _ in range(3):
        logits, _ = model(tokens, copy_source_mask=source_mask)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(int(next_token.item()))
        tokens = torch.cat((tokens, next_token), dim=1)
        source_mask = torch.cat(
            (source_mask, torch.zeros_like(next_token, dtype=torch.bool)),
            dim=1,
        )

    return tuple(generated)


def main() -> None:
    seed_everything(1337)
    device = resolve_device("auto")
    rows = [
        example(left, right)
        for index, (left, right) in enumerate(zip(FIRST, SECOND))
        for other, right in enumerate(SECOND)
        if index != other
    ]
    inputs = torch.tensor([row[0] for row in rows], device=device)
    targets = torch.tensor([row[1] for row in rows], device=device)
    masks = torch.tensor([row[2] for row in rows], device=device)
    copy_targets = torch.tensor([row[3] for row in rows], device=device)
    model = build_model(
        {
            "name": "transformer",
            "embedding_dim": 32,
            "attention_heads": 4,
            "layers": 2,
            "feed_forward_dim": 64,
            "dropout": 0.0,
            "position_encoding": "rope",
            "copy_mechanism": "pointer_generator",
            "copy_loss_weight": 0.25,
        },
        vocabulary_size=64,
        block_size=16,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = None
    final = None

    for step in range(301):
        _, loss = model(
            inputs,
            targets,
            copy_source_mask=masks,
            copy_targets=copy_targets,
        )
        assert loss is not None

        if initial is None:
            initial = float(loss.detach())

        if step % 25 == 0:
            print(f"step {step:3d}: loss {float(loss.detach()):.6f}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = float(loss.detach())

    exact = 0

    for left, right in zip(FIRST, SECOND):
        generated = generate(model, left, right, device)
        expected = (left, right, END)
        matched = generated == expected
        exact += int(matched)
        print(
            f"held-out {left},{right}: generated={generated}, "
            f"expected={expected}, exact={matched}"
        )

    print(f"device: {device}")
    print(f"initial loss: {initial:.6f}")
    print(f"final loss: {final:.6f}")
    print(f"held-out exact: {exact}/8")

    if final >= 0.1 or exact != 8:
        raise SystemExit("pointer-copy overfit: failed")

    print("pointer-copy overfit: passed")


if __name__ == "__main__":
    main()
