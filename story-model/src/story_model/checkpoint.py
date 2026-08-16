"""Versioned, resumable model checkpoints.

A "checkpoint" is a snapshot of everything needed to either (a) pick up
training exactly where it left off, or (b) load a finished model for
generation/evaluation, with nothing else required. This project's
checkpoints go further than just saving model weights: they also capture
optimizer state and the random-number-generator state, which together are
what make resumed training bit-for-bit indistinguishable from training
that never stopped (see train.py's --resume path, and
restore_rng_state below).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


CHECKPOINT_VERSION = 2

EXPANDABLE_VOCABULARY_PARAMETERS = frozenset(
    {
        "token_embeddings.weight",
        "output_projection.weight",
        "output_projection.bias",
    }
)


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every random-number generator PyTorch might be using.

    Randomness shows up throughout training — weight initialization,
    which random batch get_batch() samples next, which token dropout
    zeroes out — and PyTorch tracks a separate generator's internal state
    per compute backend (CPU always; CUDA/MPS only if that hardware is
    actually available on this machine). Saving all of them (not just
    CPU) is what lets restore_rng_state() put a resumed training run into
    EXACTLY the state it would have been in if it had simply kept running
    without ever stopping — same next random batch, same next dropout
    mask, etc.
    """

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
    """Move restored AdamW state tensors to the model's device.

    AdamW (see train.py) keeps its own per-parameter running statistics
    (first and second gradient moments) as tensors inside optimizer.state
    — these are loaded from the checkpoint file onto the CPU by default
    (read_checkpoint's map_location="cpu"), so if the model itself lives
    on a GPU/MPS device, its optimizer state needs to be moved over to
    match, or the next optimizer.step() would fail trying to combine
    tensors that live on different devices.
    """

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
    """Write model (and optionally optimizer/RNG) state to a single file.

    `extra` is an open-ended dict where callers stash whatever
    self-describing metadata a checkpoint should carry — train.py uses it
    to embed the full training config and the tokenizer's own to_dict(),
    so that generate.py and evaluate.py can later reconstruct the exact
    model architecture and vocabulary a checkpoint needs, from the
    checkpoint file alone, without requiring the original config file or
    corpus to still be present on disk in the same place. CHECKPOINT_VERSION
    exists so that, if this file format ever needs to change in an
    incompatible way in the future, code can detect and handle old-format
    checkpoints explicitly rather than failing on them unpredictably.
    """

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
    # weights_only=True restricts what torch.load is willing to
    # unpickle to plain tensors/dicts/etc. — refusing to execute arbitrary
    # Python objects a malicious or corrupted checkpoint file might
    # otherwise smuggle in. It's a security-hardening default (PyTorch's
    # own recommendation for loading files from any source you don't
    # fully control) that costs nothing here, since this project never
    # needs to store arbitrary Python objects inside a checkpoint anyway.
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
    """Load a checkpoint's weights (and optionally optimizer/RNG state) in place.

    restore_rng defaults to False because it's only meaningful — and only
    safe — when resuming an INTERRUPTED training run, where reproducing
    the exact random-number sequence matters (see train.py's --resume).
    For plain inference (generate.py, evaluate.py), restoring RNG state
    would be actively wrong: those scripts want their OWN fresh,
    explicitly-chosen seed (see generate.py's `seed` argument) to control
    generation, not whatever internal state training happened to leave
    the RNG in when the checkpoint was saved.
    """

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


def load_model_warm_start(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    source_vocabulary_size: int,
    destination_vocabulary_size: int,
) -> tuple[str, ...]:
    """Load model weights while retaining new vocabulary rows.

    Unlike load_checkpoint (which requires an EXACT architecture match),
    this supports growing the vocabulary — e.g. adding Phase 15's
    character control tokens on top of an already-trained BPE vocabulary.
    Every parameter whose shape matches exactly is copied verbatim.
    Parameters in EXPANDABLE_VOCABULARY_PARAMETERS (the token embedding
    table and the output projection, both of which have one row per
    vocabulary token) are allowed to differ ONLY in their first
    dimension: the old vocabulary's rows are copied into the new,
    larger tensor's matching rows, and the new tensor's own freshly
    initialized rows (for token ids beyond source_vocabulary_size) are
    left untouched. This is what lets a model "warm start" from a
    smaller-vocabulary checkpoint — reusing everything it already
    learned — while still being ready to learn meaningful embeddings for
    the newly added tokens from scratch, without retraining the entire
    model from a random initialization.
    """

    if source_vocabulary_size < 1:
        raise ValueError(
            "source_vocabulary_size must be positive"
        )

    if destination_vocabulary_size < source_vocabulary_size:
        raise ValueError(
            "warm start cannot shrink the vocabulary"
        )

    source_state = checkpoint.get("model_state_dict")

    if not isinstance(source_state, dict):
        raise ValueError(
            "warm-start checkpoint has no model_state_dict"
        )

    destination_state = model.state_dict()
    source_keys = set(source_state)
    destination_keys = set(destination_state)

    if source_keys != destination_keys:
        missing = sorted(destination_keys - source_keys)
        unexpected = sorted(source_keys - destination_keys)
        raise ValueError(
            "warm-start model parameters do not match: "
            f"missing={missing}, unexpected={unexpected}"
        )

    adapted_state = {}
    expanded_parameters = []

    for name, destination_tensor in destination_state.items():
        source_tensor = source_state[name]

        if source_tensor.shape == destination_tensor.shape:
            adapted_state[name] = source_tensor
            continue

        can_expand = (
            name in EXPANDABLE_VOCABULARY_PARAMETERS
            and source_tensor.ndim >= 1
            and destination_tensor.ndim == source_tensor.ndim
            and source_tensor.shape[0] == source_vocabulary_size
            and destination_tensor.shape[0]
            == destination_vocabulary_size
            and source_tensor.shape[1:]
            == destination_tensor.shape[1:]
        )

        if not can_expand:
            raise ValueError(
                "warm-start parameter shape mismatch for "
                f"{name}: checkpoint {tuple(source_tensor.shape)}, "
                f"model {tuple(destination_tensor.shape)}"
            )

        # Clone rather than mutate destination_tensor in place: it's
        # already been randomly initialized (including the "new" rows
        # for the added vocabulary), and clone() preserves that
        # initialization for the rows we're not overwriting below.
        expanded_tensor = destination_tensor.clone()
        expanded_tensor[:source_vocabulary_size].copy_(
            source_tensor
        )
        adapted_state[name] = expanded_tensor
        expanded_parameters.append(name)

    model.load_state_dict(adapted_state, strict=True)
    return tuple(expanded_parameters)
