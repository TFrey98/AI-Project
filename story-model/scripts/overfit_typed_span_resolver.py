"""Primitive proof for typed routing and unseen candidate combinations."""

from __future__ import annotations

import argparse

import torch

from story_model.data import ByteBPETokenizer
from story_model.models import build_model
from story_model.runtime import resolve_device, seed_everything
from story_model.typed_span_resolver import (
    CLARIFICATION_RESPONSE,
    CLARIFY_ACTION,
    GENERATE_ACTION,
    RESOLVE_ACTION,
    RESPONSE_FRAMES,
    TYPED_SPAN_CONTROL_TOKENS,
    TypedSpanCandidate,
    TypedSpanRecord,
    TypedSpanResolver,
    encode_typed_span_records,
    realize_resolver_decision,
    typed_span_batch,
)


PRIMITIVE_VALUES = (
    "alder bloom",
    "birch bronze",
    "cinder coral",
    "dusk cream",
    "ember flame",
    "fern haze",
    "flint jade",
    "frost lavender",
    "granite mist",
    "harbor plum",
    "hazel rust",
    "iron saffron",
    "juniper smoke",
    "linden umber",
    "moss wine",
    "pearl yellow",
)
TRAIN_PAIRS = tuple(
    (value, PRIMITIVE_VALUES[(index + 1) % len(PRIMITIVE_VALUES)])
    for index, value in enumerate(PRIMITIVE_VALUES)
)
HELD_OUT_PAIRS = (
    ("alder bloom", "fern haze"),
    ("birch bronze", "flint jade"),
    ("cinder coral", "frost lavender"),
    ("dusk cream", "granite mist"),
)


def _resolve(pair, side: int, record_id: str, split: str):
    expected = pair[side]
    alternative = pair[1 - side]
    return TypedSpanRecord(
        record_id=record_id,
        source_context_id=record_id,
        conversation_id=record_id.rsplit(":", 1)[0],
        split=split,
        skill="supplied_fact",
        case="supported",
        prompt=(
            f"Verified evidence: the sample color is {expected}. "
            f"A superseded note claimed {alternative}. "
            "Question: which color does the verified evidence identify?"
        ),
        expected_action=RESOLVE_ACTION,
        expected_type="color",
        candidates=tuple(
            TypedSpanCandidate(value, "color") for value in pair
        ),
        selected_candidate_index=side,
        response_template=RESPONSE_FRAMES["supplied_fact"],
        expected_value=expected,
        alternative_value=alternative,
    )


def _records(pairs, split: str):
    records = []
    for index, pair in enumerate(pairs):
        conversation = f"{split}:{index:03d}"
        records.extend(
            (
                _resolve(pair, 0, f"{conversation}:0", split),
                _resolve(pair, 1, f"{conversation}:1", split),
                TypedSpanRecord(
                    record_id=f"{conversation}:clarify",
                    source_context_id=f"{conversation}:clarify",
                    conversation_id=f"{conversation}:clarify",
                    split=split,
                    skill="supplied_fact",
                    case="missing_evidence",
                    prompt=(
                        "No verified color evidence is available. "
                        "Question: which color does the evidence identify?"
                    ),
                    expected_action=CLARIFY_ACTION,
                    expected_type="color",
                    candidates=tuple(
                        TypedSpanCandidate(value, "color") for value in pair
                    ),
                    selected_candidate_index=None,
                    response_template=CLARIFICATION_RESPONSE,
                    expected_value=pair[0],
                    alternative_value=pair[1],
                ),
                TypedSpanRecord(
                    record_id=f"{conversation}:generate",
                    source_context_id=f"{conversation}:generate",
                    conversation_id=f"{conversation}:generate",
                    split=split,
                    skill="supplied_fact",
                    case="ordinary_generation",
                    prompt="Acknowledge that you heard the request.",
                    expected_action=GENERATE_ACTION,
                    expected_type=None,
                    candidates=(),
                    selected_candidate_index=None,
                    response_template=None,
                    expected_value=None,
                    alternative_value=None,
                ),
            )
        )
    return tuple(records)


@torch.no_grad()
def _assess(model, examples, records, device):
    batch = typed_span_batch(examples, range(len(examples)), device)
    output = model(*batch[:5])
    actions = output.action_logits.argmax(dim=-1).tolist()
    candidates = output.candidate_logits.argmax(dim=-1).tolist()
    structured = 0
    exact_resolve = 0
    resolve_total = 0
    for record, action_index, candidate_index in zip(
        records, actions, candidates
    ):
        action = (RESOLVE_ACTION, CLARIFY_ACTION, GENERATE_ACTION)[action_index]
        action_ok = action == record.expected_action
        candidate_ok = (
            record.expected_action != RESOLVE_ACTION
            or candidate_index == record.selected_candidate_index
        )
        structured += int(action_ok and candidate_ok)
        if record.expected_action == RESOLVE_ACTION:
            resolve_total += 1
            decision = realize_resolver_decision(
                record, action, candidate_index
            )
            exact_resolve += int(
                decision.action == RESOLVE_ACTION
                and decision.selected_value == record.expected_value
                and record.expected_value in (decision.text or "")
            )
    return structured / len(records), exact_resolve, resolve_total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    seed_everything(1337)
    device = torch.device(resolve_device(args.device))
    training_records = _records(TRAIN_PAIRS, "train")
    held_out_records = _records(HELD_OUT_PAIRS, "val")
    tokenizer_text = "\n".join(
        record.prompt for record in training_records
    )
    tokenizer = ByteBPETokenizer.train(
        tokenizer_text, vocab_size=288, min_frequency=2
    ).with_special_tokens(TYPED_SPAN_CONTROL_TOKENS)
    block_size = 192
    training_examples = encode_typed_span_records(
        training_records, tokenizer, block_size
    )
    held_out_examples = encode_typed_span_records(
        held_out_records, tokenizer, block_size
    )
    backbone = build_model(
        {
            "name": "transformer",
            "position_encoding": "rope",
            "normalization": "layernorm",
            "feed_forward_activation": "swiglu",
            "embedding_dim": 32,
            "attention_heads": 4,
            "layers": 1,
            "feed_forward_dim": 64,
            "dropout": 0.0,
        },
        tokenizer.vocab_size,
        block_size,
    )
    model = TypedSpanResolver(backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3)
    batch = typed_span_batch(
        training_examples, range(len(training_examples)), device
    )
    initial = None
    final = None
    for step in range(args.steps + 1):
        output = model(*batch[:5], batch[5], batch[6])
        assert output.loss is not None
        if initial is None:
            initial = float(output.loss.detach())
        final = float(output.loss.detach())
        if step % 25 == 0:
            print(f"step {step:3d}: loss {final:.6f}")
        if step == args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    train_accuracy, _, _ = _assess(
        model, training_examples, training_records, device
    )
    held_accuracy, held_exact, held_total = _assess(
        model, held_out_examples, held_out_records, device
    )
    print(f"device: {device}")
    print(f"initial loss: {initial:.6f}")
    print(f"final loss: {final:.6f}")
    print(f"train structured accuracy: {train_accuracy:.3f}")
    print(f"held-out exact resolution: {held_exact}/{held_total}")
    print(f"held-out structured accuracy: {held_accuracy:.3f}")
    if train_accuracy != 1.0 or held_exact != held_total or held_accuracy != 1.0:
        raise SystemExit("typed-span resolver overfit: failed")
    print("typed-span resolver overfit: passed")


if __name__ == "__main__":
    main()
