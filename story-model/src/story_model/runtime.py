"""Runtime helpers shared by training and generation.

Small, easy-to-overlook utilities that matter a lot in practice: which
accelerator (CPU/CUDA/MPS) actually runs the model, making randomness
reproducible, and correctly measuring memory/timing on each backend.
Centralizing them here means train.py, generate.py, and evaluate.py all
handle "which device" and "how do I seed everything" identically, instead
of each subtly reimplementing (and potentially getting slightly wrong)
the same device-detection logic.
"""

import torch


def resolve_device(requested: str = "auto") -> str:
    """Turn a config's device string into a concrete, available device.

    "auto" picks the best available accelerator in priority order — CUDA
    (NVIDIA GPUs) first, then MPS (Apple Silicon GPUs, what this project
    actually runs on during development), falling back to plain CPU if
    neither is present. This is what lets the exact same config file train
    correctly on very different machines without editing it — a config
    checked into git shouldn't need to hardcode one developer's specific
    hardware. Explicitly requesting "cuda" or "mps" on a machine that
    doesn't have it fails loudly and immediately, rather than silently
    falling back to a much slower CPU run that someone might not notice
    for a long time.
    """

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
    """Seed every accelerator available to this PyTorch process.

    A model's training run isn't governed by ONE random number generator
    — CPU tensor operations, CUDA operations, and MPS operations each draw
    from their own separate stream, and a script might use any
    combination of them. Seeding only torch.manual_seed(seed) (the CPU
    generator) would leave GPU-side randomness — including weight
    initialization and dropout, if any layer's computation happens to run
    on the accelerator — uncontrolled, silently breaking reproducibility
    on any machine with a GPU. This function is the one place all of
    training's "make results reproducible" logic lives.
    """

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def synchronize_device(device: str) -> None:
    """Wait for queued accelerator operations to complete.

    GPU/MPS operations are launched ASYNCHRONOUSLY by default — a Python
    line that "runs" a matrix multiply on the accelerator returns
    immediately, before the hardware has actually finished the work, so
    the CPU can keep issuing more instructions without stalling on every
    single op. That's great for throughput, but it means any CPU-side
    timing measurement (train.py wraps its tokens/second calculation with
    calls to this function) would otherwise measure "how long did it take
    to QUEUE the work" rather than "how long did the work actually take."
    Synchronizing forces a wait until every previously-queued operation on
    that device has genuinely finished, so timing measurements around it
    are accurate. (Plain CPU execution is already synchronous, so there's
    nothing to do in that case.)
    """

    if device == "cuda":
        torch.cuda.synchronize()

    elif device == "mps":
        torch.mps.synchronize()


def current_memory_bytes(device: str) -> int:
    """Return memory currently occupied by tensors.

    Used by train.py to report peak memory usage during training — a
    practical signal for whether a given model/batch-size configuration
    would fit on a given machine before committing to a long run. Plain
    CPU memory isn't tracked the same way GPU/MPS allocator memory is (no
    dedicated PyTorch API for it here), so this returns 0 for CPU rather
    than a possibly-misleading approximation.
    """

    if device == "cuda":
        return torch.cuda.memory_allocated()

    if device == "mps":
        return torch.mps.current_allocated_memory()

    return 0
