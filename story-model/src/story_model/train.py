"""Train configured language models and save checkpoints.

This is the project's main training loop: read a YAML config, build a
tokenizer and model from it, then repeatedly (1) sample a random batch of
text, (2) measure how wrong the model's next-token predictions are, and
(3) nudge every weight slightly in the direction that would have made
those predictions less wrong. Do that tens of thousands of times and the
model gradually gets better at predicting text like its training corpus.
Nothing here is Transformer-specific — the same loop trains the bigram,
attention, and full Transformer models in this project identically,
because "how do you train a model" and "what does the model compute" are
deliberately separate concerns (see story_model.models.build_model for the
architecture side of that split).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import yaml

from story_model.checkpoint import (
    load_checkpoint,
    load_model_warm_start,
    read_checkpoint,
    save_checkpoint,
)
from story_model.data import (
    ByteBPETokenizer,
    build_tokenizer,
    get_batch,
    load_corpus_manifest,
    load_text_splits,
    tokenizer_from_dict,
)
from story_model.models import build_model
from story_model.runtime import (
    current_memory_bytes,
    resolve_device,
    seed_everything,
    synchronize_device,
)


def learning_rate_for_step(
    step: int,
    max_steps: int,
    maximum_rate: float,
    minimum_rate: float,
    warmup_steps: int,
) -> float:
    """Linear warmup followed by cosine decay.

    This is the de facto standard learning-rate schedule for training
    Transformers, and it solves two separate problems:

    WARMUP (the first `warmup_steps` steps, ramping linearly from ~0 up to
    `maximum_rate`): at the very start of training, the model's weights
    are freshly randomly initialized, so its first gradient estimates are
    large, noisy, and not yet trustworthy. Taking a full-size optimizer
    step on that noisy signal risks a bad update that the model struggles
    to recover from. Starting with a tiny learning rate and ramping up
    lets the model's early weights settle into a reasonable region before
    trusting full-size updates.

    DECAY (a cosine curve from `maximum_rate` down to `minimum_rate` over
    the rest of training): early in training, large steps are good — they
    make fast progress while the model is still very wrong. Late in
    training, large steps are bad — they cause the loss to bounce around
    near its best value instead of settling into it. A smooth cosine
    decay (rather than a sudden drop) gradually shifts from "move fast"
    to "fine-tune carefully" as training progresses. (A cosine curve is
    used, rather than e.g. a straight line, mainly because it decelerates
    the rate change near both ends — this schedule was popularized by
    "SGDR: Stochastic Gradient Descent with Warm Restarts" and is standard
    practice for LLM pretraining.)
    """

    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative")

    if warmup_steps >= max_steps:
        raise ValueError(
            "warmup_steps must be smaller than max_steps"
        )

    if warmup_steps > 0 and step < warmup_steps:
        return maximum_rate * (
            (step + 1) / warmup_steps
        )

    decay_steps = max_steps - warmup_steps

    if decay_steps <= 1:
        return minimum_rate

    decay_position = (
        step - warmup_steps
    ) / (decay_steps - 1)

    decay_position = min(
        max(decay_position, 0.0),
        1.0,
    )

    # cos ranges from 1 (at decay_position=0) down to -1 (at
    # decay_position=1); rescaling to (0.5 * (1 + cos)) maps that to a
    # smooth 1 -> 0 curve, which is then interpolated between
    # maximum_rate and minimum_rate.
    cosine = 0.5 * (
        1.0 + math.cos(math.pi * decay_position)
    )

    return minimum_rate + cosine * (
        maximum_rate - minimum_rate
    )


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    # PyTorch optimizers store their learning rate per "parameter group"
    # (a mechanism for giving different subsets of parameters different
    # hyperparameters) rather than as one single attribute — this project
    # doesn't use multiple parameter groups, but the loop still has to
    # write to all of them to change the effective rate.
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def validation_loss_improved(
    validation_loss: float,
    best_validation_loss: float | None,
) -> bool:
    """Return whether a finite validation loss is a strict improvement.

    Two guards are bundled into this one predicate, both there to prevent
    best.pt (see save_best_validation_checkpoint) from ever being
    overwritten with something worse than what it already holds:
    math.isfinite() rejects NaN/Inf losses (which can occur transiently
    from numerical instability and would otherwise corrupt the "best"
    tracking — NaN compares False to everything, so a naive `<` check
    alone wouldn't reliably exclude it), and the strict `<` (not `<=`)
    means a validation loss that TIES the current best doesn't trigger a
    redundant checkpoint write.
    """

    return (
        math.isfinite(validation_loss)
        and (
            best_validation_loss is None
            or validation_loss < best_validation_loss
        )
    )


def save_best_validation_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_metadata: dict,
    validation_loss: float,
) -> None:
    """Save a resumable checkpoint taken immediately after evaluation.

    Why keep a separate best.pt alongside final.pt at all: training loss
    generally keeps improving the longer you train (the model can always
    fit its training data a little better), but validation loss can start
    getting WORSE partway through if the model begins overfitting —
    memorizing training-specific quirks instead of learning generalizable
    patterns. final.pt always reflects the model after the complete
    configured step budget, which is useful for comparing configs
    apples-to-apples, but best.pt is what you actually want to *use* if
    your goal is the best-generalizing model, since it's the checkpoint
    take at the single update where held-out performance was highest.
    """

    # dict(checkpoint_metadata) makes a shallow copy so this function can
    # freely add best.pt-specific keys (below) without mutating the
    # metadata dict that final.pt and future step_N.pt checkpoints share.
    best_metadata = dict(checkpoint_metadata)
    best_metadata.update(
        {
            "best_validation_loss": validation_loss,
            "best_step": step,
            # Recorded so that if training is interrupted and resumed
            # from this checkpoint, the loop below (see
            # `evaluation_already_completed`) knows evaluation at this
            # exact step already ran and its result already got recorded
            # — without this, resuming exactly at an eval step could
            # double-count that step's evaluation.
            "evaluation_completed_step": step,
        }
    )

    save_checkpoint(
        path,
        model,
        optimizer,
        step,
        extra=best_metadata,
    )


def tokenizer_for_warm_start(
    checkpoint: dict,
    tokenizer_config: dict | None,
) -> tuple[ByteBPETokenizer, ByteBPETokenizer]:
    """Extend a checkpoint BPE tokenizer without changing old ids.

    Returns (source_tokenizer, destination_tokenizer): the tokenizer the
    checkpoint was actually trained with, and a new tokenizer sharing the
    exact same merges plus whatever additional special_tokens the current
    config asks for. The strict checks here exist because warm-starting
    only makes sense — and load_model_warm_start's row-copying trick (see
    checkpoint.py) is only correct — if every token id the OLD model
    already learned means EXACTLY the same thing in the new tokenizer.
    Retraining or resizing the base BPE vocabulary, or reordering already-
    assigned special tokens, would silently break that guarantee, so both
    are rejected outright rather than allowed to produce a model that
    trains "successfully" on a corrupted vocabulary mapping.
    """

    tokenizer_data = (
        checkpoint.get("extra", {}).get("tokenizer")
    )

    if tokenizer_data is None:
        raise ValueError(
            "warm-start checkpoint has no tokenizer metadata"
        )

    source_tokenizer = tokenizer_from_dict(tokenizer_data)

    if not isinstance(source_tokenizer, ByteBPETokenizer):
        raise ValueError(
            "character warm start requires a byte-BPE tokenizer"
        )

    tokenizer_config = tokenizer_config or {}

    if tokenizer_config.get("type", "byte_bpe") != "byte_bpe":
        raise ValueError(
            "warm-start tokenizer type must be byte_bpe"
        )

    configured_base_size = int(
        tokenizer_config.get(
            "vocab_size",
            source_tokenizer.base_vocab_size,
        )
    )

    if configured_base_size != source_tokenizer.base_vocab_size:
        raise ValueError(
            "warm start cannot retrain or resize the base BPE "
            "vocabulary"
        )

    configured_special_tokens = tuple(
        tokenizer_config.get(
            "special_tokens",
            source_tokenizer.special_tokens,
        )
    )

    existing_count = len(source_tokenizer.special_tokens)

    # New special tokens may only be APPENDED, never reordered or
    # inserted before existing ones — a special token's id is simply its
    # position in this list (see ByteBPETokenizer.__post_init__), so
    # changing the order of already-assigned tokens would silently
    # reassign their ids out from under a model that already learned
    # embeddings for the old ids.
    if (
        configured_special_tokens[:existing_count]
        != source_tokenizer.special_tokens
    ):
        raise ValueError(
            "warm start must preserve existing special-token ids"
        )

    destination_tokenizer = ByteBPETokenizer(
        merges=list(source_tokenizer.merges),
        special_tokens=configured_special_tokens,
    )

    return source_tokenizer, destination_tokenizer


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    """Estimate train/val loss by averaging over several random batches.

    This deliberately does NOT compute the exact loss over the entire
    training or validation set (unlike evaluate.py's evaluate_token_stream,
    which does exactly that, once, for final reporting) — doing so every
    time training pauses to check progress would be far too slow to run
    every `eval_interval` steps. Instead it samples `eval_iters` random
    batches (the same get_batch() used for training itself) and averages
    their loss. This is a Monte Carlo estimate: noisier than the exact
    figure, but cheap enough to compute frequently, which matters more for
    tracking whether training is progressing than getting the number
    exactly right at every checkpoint.

    model.eval() / model.train() around the measurement matter because
    some layers (dropout, batch norm — this project's models use dropout)
    behave differently during training vs. evaluation: dropout randomly
    zeroes activations during training (a regularization technique that
    forces the network not to over-rely on any single neuron) but should
    be switched OFF for evaluation, so the measurement reflects the
    model's actual, undistorted predictions.
    """

    model.eval()
    output = {}

    for split, data in (
        ("train", train_data),
        ("val", val_data),
    ):
        losses = torch.zeros(eval_iters)

        for index in range(eval_iters):
            inputs, targets = get_batch(
                data,
                block_size,
                batch_size,
                device,
            )

            _, loss = model(inputs, targets)

            if loss is None:
                raise RuntimeError(
                    "Model did not return evaluation loss"
                )

            losses[index] = loss.item()

        output[split] = losses.mean().item()

    model.train()
    return output


def train(
    config_path: str,
    resume_path: str | None = None,
    warm_start_path: str | None = None,
) -> None:
    if resume_path is not None and warm_start_path is not None:
        raise ValueError(
            "resume_path and warm_start_path are mutually exclusive"
        )

    config = yaml.safe_load(
        Path(config_path).read_text(
            encoding="utf-8"
        )
    )

    train_config = config["train"]
    data_config = config["data"]
    checkpoint_config = config["checkpoint"]

    max_steps = int(train_config["max_steps"])
    maximum_rate = float(
        train_config["learning_rate"]
    )
    minimum_rate = float(
        train_config.get(
            "min_learning_rate",
            maximum_rate,
        )
    )
    warmup_steps = int(
        train_config.get("warmup_steps", 0)
    )
    gradient_clip = float(
        train_config.get("gradient_clip", 0.0)
    )
    log_interval = int(
        train_config.get(
            "log_interval",
            train_config["eval_interval"],
        )
    )
    weight_decay = float(
        train_config.get("weight_decay", 0.01)
    )

    if log_interval < 1:
        raise ValueError(
            "log_interval must be positive"
        )

    # Seeding BEFORE any random operation (batch sampling, weight
    # initialization, dropout) is what makes two runs of the same config
    # produce bit-for-bit identical results — essential both for debugging
    # ("did my code change actually change behavior, or is this just
    # random variance?") and for the controlled ablations this project
    # runs between architecture variants (RoPE vs. learned position
    # encoding, GELU vs. SwiGLU, etc.), where the whole point is to
    # isolate ONE variable rather than let random seed differences muddy
    # the comparison.
    seed_everything(train_config["seed"])
    device = resolve_device(train_config["device"])

    training_text, validation_text = load_text_splits(
        data_config
    )

    resume_checkpoint = None
    warm_start_checkpoint = None
    source_tokenizer = None

    if resume_path is not None:
        # When resuming, the tokenizer MUST come from the checkpoint,
        # never be rebuilt from the corpus — a token id only means
        # anything relative to the specific vocabulary it was trained
        # with, so rebuilding (especially for BPE, whose merge list
        # depends on corpus statistics) risks silently producing a
        # DIFFERENT vocabulary that happens to have the same size but maps
        # ids to different tokens, which would corrupt every embedding the
        # model already learned.
        resume_checkpoint = read_checkpoint(
            resume_path,
            map_location="cpu",
        )
        tokenizer_data = (
            resume_checkpoint
            .get("extra", {})
            .get("tokenizer")
        )
        tokenizer = (
            tokenizer_from_dict(tokenizer_data)
            if tokenizer_data is not None
            else build_tokenizer(
                training_text=training_text,
                config=config.get("tokenizer"),
            )
        )
    elif warm_start_path is not None:
        # A warm start's tokenizer must EXTEND (not replace) the source
        # checkpoint's vocabulary — see tokenizer_for_warm_start's
        # docstring for why reusing old token ids exactly is what makes
        # the weight-copying trick in load_model_warm_start correct.
        warm_start_checkpoint = read_checkpoint(
            warm_start_path,
            map_location="cpu",
        )
        source_tokenizer, tokenizer = tokenizer_for_warm_start(
            warm_start_checkpoint,
            config.get("tokenizer"),
        )
    else:
        tokenizer = build_tokenizer(
            training_text=training_text,
            config=config.get("tokenizer"),
        )

    checkpoint_metadata = {
        "config": config,
        "tokenizer": tokenizer.to_dict(),
    }
    corpus_manifest = load_corpus_manifest(data_config)

    if corpus_manifest is not None:
        checkpoint_metadata["corpus_manifest"] = (
            corpus_manifest
        )

    if warm_start_checkpoint is not None:
        assert source_tokenizer is not None
        # Recorded in the new checkpoint's own metadata so its provenance
        # (which checkpoint it warm-started from, and exactly how the
        # vocabulary grew) is traceable later, the same self-describing
        # philosophy as the rest of checkpoint_metadata.
        checkpoint_metadata["warm_start"] = {
            "checkpoint": str(warm_start_path),
            "source_step": int(
                warm_start_checkpoint.get("step", 0)
            ),
            "source_vocabulary_size": (
                source_tokenizer.vocab_size
            ),
            "destination_vocabulary_size": (
                tokenizer.vocab_size
            ),
        }

    train_data = torch.tensor(
        tokenizer.encode(training_text),
        dtype=torch.long,
    )
    val_data = torch.tensor(
        tokenizer.encode(validation_text),
        dtype=torch.long,
    )

    model = build_model(
        config["model"],
        vocabulary_size=tokenizer.vocab_size,
        block_size=data_config["block_size"],
    )

    expanded_parameters: tuple[str, ...] = ()

    if warm_start_checkpoint is not None:
        assert source_tokenizer is not None
        saved_config = (
            warm_start_checkpoint
            .get("extra", {})
            .get("config")
        )

        # Same guard as the --resume path below: warm-starting into an
        # architecture that doesn't match what the source checkpoint was
        # trained with would otherwise load_model_warm_start's shape
        # checks into a confusing low-level error instead of a clear one.
        if (
            saved_config is not None
            and saved_config["model"] != config["model"]
        ):
            raise ValueError(
                "Warm-start model configuration does not match "
                "the current configuration"
            )

        expanded_parameters = load_model_warm_start(
            model=model,
            checkpoint=warm_start_checkpoint,
            source_vocabulary_size=(
                source_tokenizer.vocab_size
            ),
            destination_vocabulary_size=tokenizer.vocab_size,
        )

        checkpoint_metadata["warm_start"][
            "expanded_parameters"
        ] = list(expanded_parameters)

    model = model.to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    # AdamW: Adam with "decoupled" weight decay. Plain Adam adds an L2
    # penalty into the gradient itself before Adam's own adaptive-learning-
    # rate math is applied, which (somewhat counterintuitively) makes the
    # penalty's effective strength depend on Adam's per-parameter
    # adaptive scaling. AdamW instead applies weight decay as a separate,
    # direct shrinkage of the weights each step, independent of Adam's
    # gradient adaptation. This decoupling was shown (in "Decoupled Weight
    # Decay Regularization") to work better in practice and is the
    # standard optimizer for training Transformers. weight_decay itself is
    # a regularizer — it continuously pulls every weight a little toward
    # zero, which discourages the model from relying on any single weight
    # growing very large and helps it generalize rather than overfit.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=maximum_rate,
        weight_decay=weight_decay,
    )

    start_step = 0
    best_validation_loss = None
    best_step = None
    evaluation_completed_step = None

    if resume_path is not None:
        # restore_rng=True restores not just the model/optimizer weights
        # but also PyTorch's internal random-number-generator state at the
        # moment the checkpoint was saved — the same reproducibility
        # motivation as seed_everything() above, extended across a
        # stop/resume boundary: without this, resumed training would
        # sample a DIFFERENT sequence of random batches than an
        # uninterrupted run of the same config would have, even with the
        # same starting seed, because the RNG's internal state had already
        # advanced partway through that sequence before the checkpoint was
        # taken.
        checkpoint = load_checkpoint(
            resume_path,
            model,
            optimizer,
            map_location="cpu",
            restore_rng=True,
        )

        start_step = int(
            checkpoint.get("step", 0)
        )

        checkpoint_extra = checkpoint.get("extra", {})
        saved_best_validation_loss = checkpoint_extra.get(
            "best_validation_loss"
        )

        if saved_best_validation_loss is not None:
            best_validation_loss = float(
                saved_best_validation_loss
            )

        saved_best_step = checkpoint_extra.get("best_step")

        if saved_best_step is not None:
            best_step = int(saved_best_step)

        saved_evaluation_step = checkpoint_extra.get(
            "evaluation_completed_step"
        )

        if saved_evaluation_step is not None:
            evaluation_completed_step = int(
                saved_evaluation_step
            )

        saved_config = (
            checkpoint
            .get("extra", {})
            .get("config")
        )

        # Resuming with a DIFFERENT model architecture than the checkpoint
        # was trained with would silently produce garbage: load_checkpoint
        # above would either error trying to load mismatched-shape weights
        # into the model, or (worse, if shapes happened to coincidentally
        # match) load weights that mean something completely different
        # under the new architecture. Checking explicitly gives a clear
        # error message instead of a confusing crash or, worse, silently
        # wrong behavior.
        if (
            saved_config is not None
            and saved_config["model"] != config["model"]
        ):
            raise ValueError(
                "Checkpoint model configuration "
                "does not match the current configuration"
            )

        if start_step >= max_steps:
            raise ValueError(
                f"Checkpoint has completed {start_step} steps, "
                f"but max_steps is {max_steps}"
            )

        print(
            f"resuming from: {resume_path}"
        )
        print(
            f"completed steps: {start_step}"
        )

    else:
        # Model construction consumes a different number of random
        # values for different architectures. Reset before evaluation
        # and training so matched experiments see identical batches
        # and dropout streams.
        seed_everything(train_config["seed"])

    if best_validation_loss is not None:
        checkpoint_metadata["best_validation_loss"] = (
            best_validation_loss
        )

    if best_step is not None:
        checkpoint_metadata["best_step"] = best_step

    if warm_start_checkpoint is not None:
        print(f"warm starting from: {warm_start_path}")
        print(
            "source completed steps: "
            f"{warm_start_checkpoint.get('step', 0)}"
        )
        print(
            "expanded vocabulary parameters: "
            + ", ".join(expanded_parameters)
        )

    print(f"device: {device}")
    print(f"parameters: {parameter_count:,}")
    print(f"tokenizer: {tokenizer.to_dict()['type']}")
    print(f"vocabulary: {tokenizer.vocab_size}")
    print(
        f"training tokens: {len(train_data):,}"
    )
    print(
        f"validation tokens: {len(val_data):,}"
    )
    print(
        "training UTF-8 bytes: "
        f"{len(training_text.encode('utf-8')):,}"
    )
    print(
        "validation UTF-8 bytes: "
        f"{len(validation_text.encode('utf-8')):,}"
    )
    # bytes/token reports the tokenizer's compression ratio directly in
    # the training log — a quick sanity check that, e.g., a BPE run really
    # is packing more information per token than a char-level run would
    # (~1.0 bytes/token for ASCII text), without needing to run the
    # separate scripts/inspect_bpe.py tool.
    print(
        "training bytes/token: "
        f"{len(training_text.encode('utf-8')) / len(train_data):.3f}"
    )
    print(
        "validation bytes/token: "
        f"{len(validation_text.encode('utf-8')) / len(val_data):.3f}"
    )

    block_size = data_config["block_size"]
    batch_size = data_config["batch_size"]
    eval_interval = train_config["eval_interval"]
    eval_iters = train_config["eval_iters"]
    save_interval = checkpoint_config["save_interval"]
    best_path = (
        Path(checkpoint_config["dir"])
        / "best.pt"
    )

    peak_memory = current_memory_bytes(device)
    tokens_since_log = 0

    synchronize_device(device)
    timing_started = time.perf_counter()

    last_loss = None
    last_gradient_norm = None

    for step in range(start_step, max_steps):
        learning_rate = learning_rate_for_step(
            step=step,
            max_steps=max_steps,
            maximum_rate=maximum_rate,
            minimum_rate=minimum_rate,
            warmup_steps=warmup_steps,
        )

        set_learning_rate(
            optimizer,
            learning_rate,
        )

        # If resuming exactly at a step where evaluation had already run
        # (and been recorded via save_best_validation_checkpoint's
        # "evaluation_completed_step"), skip re-running it — otherwise a
        # resume landing precisely on an eval_interval boundary would
        # evaluate that step twice and could spuriously "improve" on
        # itself in the best.pt log output.
        evaluation_already_completed = (
            evaluation_completed_step == step
        )

        if (
            step % eval_interval == 0
            and not evaluation_already_completed
        ):
            synchronize_device(device)

            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                block_size=block_size,
                batch_size=batch_size,
                eval_iters=eval_iters,
                device=device,
            )

            synchronize_device(device)

            print(
                f"update {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, "
                f"lr {learning_rate:.6g}"
            )

            if validation_loss_improved(
                losses["val"],
                best_validation_loss,
            ):
                best_validation_loss = losses["val"]
                best_step = step
                checkpoint_metadata[
                    "best_validation_loss"
                ] = best_validation_loss
                checkpoint_metadata["best_step"] = best_step

                save_best_validation_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    checkpoint_metadata=checkpoint_metadata,
                    validation_loss=best_validation_loss,
                )

                print(
                    "new best: "
                    f"val loss {best_validation_loss:.4f} "
                    f"at update {best_step}"
                )

            # Exclude evaluation time from throughput.
            tokens_since_log = 0
            timing_started = time.perf_counter()

        if evaluation_already_completed:
            evaluation_completed_step = None

        inputs, targets = get_batch(
            train_data,
            block_size,
            batch_size,
            device,
        )

        _, loss = model(inputs, targets)

        if loss is None:
            raise RuntimeError(
                "Model did not return training loss"
            )

        # The four-step heart of gradient descent: clear old gradients,
        # compute new ones via backpropagation, optionally clip them, then
        # apply the update.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if gradient_clip > 0.0:
            # Gradient clipping rescales the ENTIRE gradient vector (across
            # all parameters at once) if its overall norm exceeds
            # gradient_clip, while preserving its direction. This is a
            # safety net against occasional "gradient spikes" — batches
            # that happen to produce an unusually large gradient, which,
            # left unclipped, could take a step so large it knocks the
            # model into a much worse region of weight-space (or, in
            # extreme cases, produces NaN/Inf weights that never recover).
            # It's especially important early in training and for deep
            # stacks of layers, where gradients can compound across many
            # blocks.
            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip,
                )
            )
        else:
            gradient_norm = torch.tensor(
                float("nan")
            )

        optimizer.step()

        last_loss = loss
        last_gradient_norm = gradient_norm
        tokens_since_log += inputs.numel()

        peak_memory = max(
            peak_memory,
            current_memory_bytes(device),
        )

        completed_steps = step + 1

        if completed_steps % log_interval == 0:
            synchronize_device(device)

            elapsed = (
                time.perf_counter()
                - timing_started
            )

            tokens_per_second = (
                tokens_since_log / elapsed
                if elapsed > 0
                else 0.0
            )

            memory_mebibytes = (
                current_memory_bytes(device)
                / (1024**2)
            )

            gradient_value = (
                last_gradient_norm.item()
                if last_gradient_norm is not None
                else float("nan")
            )

            print(
                f"step {completed_steps}: "
                f"loss {last_loss.item():.4f}, "
                f"lr {learning_rate:.6g}, "
                f"grad {gradient_value:.4f}, "
                f"tokens/s {tokens_per_second:,.0f}, "
                f"memory {memory_mebibytes:.1f} MiB"
            )

            tokens_since_log = 0
            timing_started = time.perf_counter()

        if completed_steps % save_interval == 0:
            synchronize_device(device)

            save_checkpoint(
                Path(checkpoint_config["dir"])
                / f"step_{completed_steps}.pt",
                model,
                optimizer,
                completed_steps,
                extra=checkpoint_metadata,
            )

            # Exclude checkpoint writing from throughput.
            synchronize_device(device)
            tokens_since_log = 0
            timing_started = time.perf_counter()

    final_losses = estimate_loss(
        model=model,
        train_data=train_data,
        val_data=val_data,
        block_size=block_size,
        batch_size=batch_size,
        eval_iters=eval_iters,
        device=device,
    )

    final_path = (
        Path(checkpoint_config["dir"])
        / "final.pt"
    )

    if validation_loss_improved(
        final_losses["val"],
        best_validation_loss,
    ):
        best_validation_loss = final_losses["val"]
        best_step = max_steps
        checkpoint_metadata["best_validation_loss"] = (
            best_validation_loss
        )
        checkpoint_metadata["best_step"] = best_step

        save_best_validation_checkpoint(
            path=best_path,
            model=model,
            optimizer=optimizer,
            step=max_steps,
            checkpoint_metadata=checkpoint_metadata,
            validation_loss=best_validation_loss,
        )

    save_checkpoint(
        final_path,
        model,
        optimizer,
        max_steps,
        extra=checkpoint_metadata,
    )

    print(
        f"final: train loss "
        f"{final_losses['train']:.4f}, "
        f"val loss {final_losses['val']:.4f}"
    )
    print(
        f"observed peak tensor memory: "
        f"{peak_memory / (1024**2):.1f} MiB"
    )
    print(
        f"best: val loss {best_validation_loss:.4f} "
        f"at update {best_step}"
    )
    print(f"best checkpoint: {best_path}")
    print(f"checkpoint: {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to a training configuration.",
    )

    # A mutually exclusive group makes argparse itself reject
    # --resume and --warm-start together, rather than relying only on
    # train()'s own runtime check — the same constraint enforced at two
    # layers (fast CLI-level feedback plus a defensive check for any other
    # caller of train() directly).
    checkpoint_group = parser.add_mutually_exclusive_group()

    checkpoint_group.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint from which to resume.",
    )

    checkpoint_group.add_argument(
        "--warm-start",
        default=None,
        help=(
            "Optional model checkpoint whose weights initialize a "
            "new optimizer and training schedule."
        ),
    )

    args = parser.parse_args()

    train(
        config_path=args.config,
        resume_path=args.resume,
        warm_start_path=args.warm_start,
    )


if __name__ == "__main__":
    main()
