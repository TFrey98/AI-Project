"""Prove that the warm-started model can memorize one masked response."""

from __future__ import annotations

import argparse

import torch

from story_model.character_chat import generate_character_response
from story_model.character_data import (
    CHARACTER_CONTROL_TOKENS,
    load_character_context,
)
from story_model.character_training import (
    CharacterTrainingRecord,
    encode_character_training_record,
)
from story_model.checkpoint import (
    load_model_warm_start,
    read_checkpoint,
)
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.models import build_model
from story_model.runtime import resolve_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--context",
        default="examples/character_context.json",
    )
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    seed_everything(1337)
    device = resolve_device("auto")
    checkpoint = read_checkpoint(
        args.checkpoint,
        map_location="cpu",
    )
    extra = checkpoint.get("extra", {})
    tokenizer_data = extra.get("tokenizer")
    config = extra.get("config")

    if tokenizer_data is None or config is None:
        raise ValueError(
            "checkpoint must contain tokenizer and config metadata"
        )

    source_tokenizer = tokenizer_from_dict(tokenizer_data)

    if not isinstance(source_tokenizer, ByteBPETokenizer):
        raise ValueError("character overfit requires byte-BPE")

    tokenizer = source_tokenizer.with_special_tokens(
        CHARACTER_CONTROL_TOKENS
    )
    context = load_character_context(args.context)
    record = CharacterTrainingRecord(
        conversation_id=context.context_id,
        context=context,
    )
    example = encode_character_training_record(
        record,
        tokenizer,
        args.block_size,
    )

    model_config = dict(config["model"])
    model_config["dropout"] = 0.0
    model = build_model(
        model_config,
        vocabulary_size=tokenizer.vocab_size,
        block_size=args.block_size,
    )
    expanded = load_model_warm_start(
        model=model,
        checkpoint=checkpoint,
        source_vocabulary_size=source_tokenizer.vocab_size,
        destination_vocabulary_size=tokenizer.vocab_size,
    )
    model = model.to(device)

    inputs = torch.tensor(
        [example.input_ids[: example.sequence_tokens]],
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        [example.target_ids[: example.sequence_tokens]],
        dtype=torch.long,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    model.eval()

    with torch.no_grad():
        _, initial_loss_tensor = model(inputs, targets)

    assert initial_loss_tensor is not None
    initial_loss = initial_loss_tensor.item()
    print(f"step   0: loss {initial_loss:.6f}")

    model.train()

    for step in range(1, args.steps + 1):
        _, loss = model(inputs, targets)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float("inf"),
        )

        if not torch.isfinite(gradient_norm).item():
            raise RuntimeError("non-finite gradient during overfit")

        optimizer.step()

        if step % 25 == 0 or step == args.steps:
            print(f"step {step:3d}: loss {loss.item():.6f}")

    model.eval()

    with torch.no_grad():
        _, final_loss_tensor = model(inputs, targets)

    assert final_loss_tensor is not None
    final_loss = final_loss_tensor.item()

    print(f"device: {device}")
    print(f"sequence tokens: {example.sequence_tokens:,}")
    print(f"prompt tokens: {example.prompt_tokens:,}")
    print(f"supervised response tokens: {example.supervised_tokens:,}")
    print(f"complete turns dropped: {example.dropped_turns:,}")
    print(
        "expanded parameters: "
        + (", ".join(expanded) if expanded else "none")
    )
    print(f"initial loss: {initial_loss:.6f}")
    print(f"final loss: {final_loss:.6f}")

    if final_loss >= args.threshold:
        raise RuntimeError(
            "The model failed to overfit one character response"
        )

    generation = generate_character_response(
        model=model,
        tokenizer=tokenizer,
        context=context,
        block_size=args.block_size,
        device=device,
        max_new_tokens=min(
            args.block_size - 1,
            max(32, example.supervised_tokens + 16),
        ),
        seed=1337,
        greedy=True,
    )
    exact_response = generation.text == context.target_response
    end_stop = generation.stop_reason == "end"
    print(f"generated response: {generation.text}")
    print(f"exact response: {exact_response}")
    print(f"stop reason: {generation.stop_reason}")

    if not exact_response or not end_stop:
        raise RuntimeError(
            "Teacher-forced loss passed, but free-running generation failed"
        )

    print("response-only free-running overfit: passed")


if __name__ == "__main__":
    main()
