"""Primitive proof for 2--4 candidate whole-span selection."""

from __future__ import annotations

import argparse

import torch

from story_model.data import ByteBPETokenizer
from story_model.expanded_typed_span_resolver import (
    CLARIFICATION_RESPONSE,
    CLARIFY_ACTION,
    EXPANDED_CONTROL_TOKENS,
    GENERATE_ACTION,
    RESOLVE_ACTION,
    RESOLVER_ACTIONS,
    ExpandedCandidate,
    ExpandedResolverRecord,
    ExpandedTypedSpanResolver,
    encode_expanded_records,
    expanded_batch,
    realize_expanded_decision,
)
from story_model.models import build_model
from story_model.runtime import resolve_device, seed_everything


VALUES = (
    "alder alcove",
    "birch basket",
    "cinder cabinet",
    "dusk drawer",
    "ember locker",
    "fern niche",
    "flint shelf",
    "frost trunk",
    "granite coffer",
    "harbor hamper",
    "hazel crate",
    "iron cupboard",
    "juniper case",
    "linden pouch",
    "moss table",
    "pearl wardrobe",
)
TRAIN_PAIRS = tuple(
    (value, VALUES[(index + 1) % len(VALUES)])
    for index, value in enumerate(VALUES)
)
HELD_OUT_PAIRS = (
    ("alder alcove", "fern niche"),
    ("birch basket", "flint shelf"),
    ("cinder cabinet", "frost trunk"),
    ("dusk drawer", "granite coffer"),
)


def _inventory(pair, index: int, width: int):
    values = [pair[0], pair[1]]
    for value in VALUES:
        if len(values) == width:
            break
        if value not in values:
            values.append(value)
    shift = (index // 3) % width
    values = values[shift:] + values[:shift]
    return tuple(ExpandedCandidate(value, "container") for value in values)


def _records(pairs, split: str, force_width: int | None = None):
    records = []
    for index, pair in enumerate(pairs):
        width = force_width or (2 + index % 3)
        candidates = _inventory(pair, index, width)
        conversation = f"primitive_{split}_{index:03d}"
        base = {
            "source_context_id": conversation,
            "conversation_id": conversation,
            "split": split,
            "skill": "multi_turn_memory",
            "case": "supported",
            "expected_action": RESOLVE_ACTION,
            "expected_type": "container",
            "candidates": candidates,
            "response_template": "The recorded location is <|resolved_value|>.",
        }
        for side in (0, 1):
            expected = pair[side]
            alternative = pair[1 - side]
            records.append(
                ExpandedResolverRecord(
                    record_id=f"{conversation}:{side}",
                    prompt=(
                        f"Verified memory: the token is in {expected}. "
                        "Question: where is the token?"
                    ),
                    selected_candidate_index=next(
                        position
                        for position, candidate in enumerate(candidates)
                        if candidate.text == expected
                    ),
                    expected_value=expected,
                    alternative_value=alternative,
                    **base,
                )
            )
        records.append(
            ExpandedResolverRecord(
                record_id=f"{conversation}:clarify",
                source_context_id=conversation,
                conversation_id=f"{conversation}:clarify",
                split=split,
                skill="multi_turn_memory",
                case="missing_evidence",
                prompt="No verified location is available. Where is the token?",
                expected_action=CLARIFY_ACTION,
                expected_type="container",
                candidates=candidates,
                selected_candidate_index=None,
                response_template=CLARIFICATION_RESPONSE,
                expected_value=pair[0],
                alternative_value=pair[1],
            )
        )
        records.append(
            ExpandedResolverRecord(
                record_id=f"{conversation}:generate",
                source_context_id=conversation,
                conversation_id=f"{conversation}:generate",
                split=split,
                skill="multi_turn_memory",
                case="ordinary_generation",
                prompt="Acknowledge that you heard the request.",
                expected_action=GENERATE_ACTION,
                expected_type=None,
                candidates=(),
                selected_candidate_index=None,
                response_template=None,
                expected_value=None,
                alternative_value=None,
            )
        )
    return tuple(records)


@torch.no_grad()
def _assess(model, examples, records, device):
    structured = 0
    exact = 0
    resolve_total = 0
    for start in range(0, len(records), 8):
        indices = range(start, min(start + 8, len(records)))
        batch_records = records[start : start + 8]
        batch = expanded_batch(examples, indices, device)
        output = model(*batch[:5])
        actions = output.action_logits.argmax(dim=-1).tolist()
        candidates = output.candidate_logits.argmax(dim=-1).tolist()
        for record, action_index, candidate_index in zip(
            batch_records, actions, candidates
        ):
            action = RESOLVER_ACTIONS[action_index]
            action_ok = action == record.expected_action
            candidate_ok = (
                record.expected_action != RESOLVE_ACTION
                or candidate_index == record.selected_candidate_index
            )
            structured += int(action_ok and candidate_ok)
            if record.expected_action == RESOLVE_ACTION:
                resolve_total += 1
                decision = realize_expanded_decision(record, action, candidate_index)
                exact += int(decision.selected_value == record.expected_value)
    return structured / len(records), exact, resolve_total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    seed_everything(1337)
    device = torch.device(resolve_device(args.device))
    train_records = _records(TRAIN_PAIRS, "train")
    held_records = _records(HELD_OUT_PAIRS, "val", force_width=4)
    tokenizer = ByteBPETokenizer.train(
        "\n".join(record.prompt for record in train_records),
        vocab_size=288,
        min_frequency=2,
    ).with_special_tokens(EXPANDED_CONTROL_TOKENS)
    block_size = 256
    train_examples = encode_expanded_records(train_records, tokenizer, block_size)
    held_examples = encode_expanded_records(held_records, tokenizer, block_size)
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
    model = ExpandedTypedSpanResolver(backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3)
    initial = None
    final = None
    for step in range(args.steps + 1):
        indices = torch.randint(0, len(train_examples), (8,)).tolist()
        batch = expanded_batch(train_examples, indices, device)
        output = model(*batch[:5], batch[5], batch[6])
        assert output.loss is not None
        if initial is None:
            initial = float(output.loss.detach())
        final = float(output.loss.detach())
        if step % 50 == 0:
            print(f"step {step:3d}: loss {final:.6f}")
        if step == args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    train_accuracy, _, _ = _assess(model, train_examples, train_records, device)
    held_accuracy, held_exact, held_total = _assess(
        model, held_examples, held_records, device
    )
    print(f"device: {device}")
    print(f"initial loss: {initial:.6f}")
    print(f"final loss: {final:.6f}")
    print(f"train structured accuracy: {train_accuracy:.3f}")
    print(f"held-out four-candidate exact resolution: {held_exact}/{held_total}")
    print(f"held-out structured accuracy: {held_accuracy:.3f}")
    if train_accuracy != 1.0 or held_exact != held_total or held_accuracy != 1.0:
        raise SystemExit("expanded typed-span resolver overfit: failed")
    print("expanded typed-span resolver overfit: passed")


if __name__ == "__main__":
    main()
