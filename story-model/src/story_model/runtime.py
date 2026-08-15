"""Runtime helpers shared by training and generation."""

import torch


def resolve_device(requested: str = "auto") -> str:
    valid_devices = {"auto", "cpu", "cuda", "mps"}

    if requested not in valid_devices:
        choices = ", ".join(sorted(valid_devices))
        raise ValueError(
            f"Unknown device {requested!r}; expected one of: {choices}"
        )

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but it is not available"
        )

    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested, but it is not available"
        )

    return requested


def seed_everything(seed: int) -> None:
    """Seed every accelerator available to this PyTorch process."""

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def synchronize_device(device: str) -> None:
    """Wait for queued accelerator operations to complete."""

    if device == "cuda":
        torch.cuda.synchronize()

    elif device == "mps":
        torch.mps.synchronize()


def current_memory_bytes(device: str) -> int:
    """Return memory currently occupied by tensors."""

    if device == "cuda":
        return torch.cuda.memory_allocated()

    if device == "mps":
        return torch.mps.current_allocated_memory()

    return 0