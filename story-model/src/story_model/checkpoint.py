"""Versioned, resumable model checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


CHECKPOINT_VERSION = 2


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()

    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()

    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if "cpu" in state:
        torch.set_rng_state(state["cpu"])

    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def move_optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move restored AdamW state tensors to the model's device."""

    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if isinstance(value, torch.Tensor):
                optimizer_state[key] = value.to(device)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "step": step,
        "extra": extra or {},
        "rng_state": capture_rng_state(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = (
            optimizer.state_dict()
        )

    torch.save(checkpoint, path)


def read_checkpoint(
    path: str | Path,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    return torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    checkpoint = read_checkpoint(
        path,
        map_location=map_location,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        model_device = next(
            model.parameters()
        ).device

        move_optimizer_to_device(
            optimizer,
            model_device,
        )

    if restore_rng and "rng_state" in checkpoint:
        restore_rng_state(
            checkpoint["rng_state"]
        )

    return checkpoint