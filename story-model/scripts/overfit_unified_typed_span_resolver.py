"""Primitive proof for count-conditioned generate/resolve/clarify routing."""

from __future__ import annotations

import argparse

import torch

from story_model.data import ByteBPETokenizer
from story_model.expanded_typed_span_resolver import (
    ACTION_TO_INDEX,
    CLARIFICATION_RESPONSE,
    CLARIFY_ACTION,
    EXPANDED_CONTROL_TOKENS,
    GENERATE_ACTION,
    MAX_CANDIDATES,
    RESOLVE_ACTION,
    ExpandedCandidate,
    ExpandedResolverRecord,
)
from story_model.models import build_model
from story_model.runtime import resolve_device, seed_everything
from story_model.unified_typed_span_resolver import (
    NO_SUPPORT_OPTION_INDEX,
    UnifiedTypedSpanResolver,
    encode_unified_records,
    realize_unified_decision,
    unified_batch,
)


TRAIN_VALUES = (
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
)
HELD_OUT_PAIRS = (
    ("juniper case", "linden pouch"),
    ("moss table", "pearl wardrobe"),
    ("quartz bureau", "raven armoire"),
    ("spruce satchel", "thistle chest"),
)


def _inventory(pair: tuple[str, str], index: int, width: int):
    values = list(pair)
    for value in TRAIN_VALUES:
        if len(values) == width:
            break
        if value not in values:
            values.append(value)
    shift = index % width
    values = values[shift:] + values[:shift]
    return tuple(ExpandedCandidate(value, "container") for value in values)


def _records(pairs, split: str, held_out: bool = False):
    records = []
    for index, pair in enumerate(pairs):
        width = 4 if held_out else 2 + index % 3
        candidates = _inventory(pair, index, width)
        conversation = f"phase33_primitive_{split}_{index:03d}"
        for side in (0, 1):
            expected = pair[side]
            alternative = pair[1 - side]
            selected = next(
                position
                for position, candidate in enumerate(candidates)
                if candidate.text == expected
            )
            records.append(
                ExpandedResolverRecord(
                    record_id=f"{conversation}:{side}",
                    source_context_id=conversation,
                    conversation_id=conversation,
                    split=split,
                    skill="multi_turn_memory",
                    case="supported",
                    prompt=(
                        f"The possible containers are {expected} and {alternative}. "
                        f"Verified memory: {alternative} is unavailable, so the "
                        f"token is in {expected}. "
                        "Question: where is the token?"
                    ),
                    expected_action=RESOLVE_ACTION,
                    expected_type="container",
                    candidates=candidates,
                    selected_candidate_index=selected,
                    response_template="The recorded location is <|resolved_value|>.",
                    expected_value=expected,
                    alternative_value=alternative,
                    source_phase="phase31",
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
                prompt=(
                    f"The possible containers are {pair[0]} and {pair[1]}. "
                    "No fact distinguishes them. Where is the token?"
                ),
                expected_action=CLARIFY_ACTION,
                expected_type="container",
                candidates=candidates,
                selected_candidate_index=None,
                response_template=CLARIFICATION_RESPONSE,
                expected_value=pair[0],
                alternative_value=pair[1],
                source_phase="phase31",
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
    model.eval()
    structured = 0
    exact = 0
    resolve_total = 0
    sentinel = 0
    clarify_total = 0
    real_candidate = 0
    multi_action = 0
    multi_action_total = 0
    for start in range(0, len(records), 8):
        indices = tuple(range(start, min(start + 8, len(records))))
        batch = unified_batch(examples, indices, device)
        output = model(*batch[:5])
        modes = output.mode_logits.argmax(dim=-1).tolist()
        options = output.option_logits.argmax(dim=-1).tolist()
        candidate_positions = torch.arange(
            MAX_CANDIDATES, device=device
        ).unsqueeze(0)
        inventory_valid = candidate_positions < batch[7].unsqueeze(1)
        real_options = (
            output.raw_option_logits[:, :MAX_CANDIDATES]
            .masked_fill(
                ~inventory_valid,
                torch.finfo(output.raw_option_logits.dtype).min,
            )
            .argmax(dim=-1)
            .tolist()
        )
        multi_actions = output.legacy_action_logits[:, :2].argmax(dim=-1).tolist()
        outcomes = zip(
            records[start : start + 8],
            modes,
            options,
            real_options,
            multi_actions,
        )
        for offset, outcome in enumerate(outcomes):
            record, mode, option, real_option, multi_action_index = outcome
            eligible_count = sum(
                examples[start + offset].option_valid_mask[:MAX_CANDIDATES]
            )
            if eligible_count >= 2 and record.expected_action != GENERATE_ACTION:
                multi_action_total += 1
                expected_multi_action = (
                    ACTION_TO_INDEX[RESOLVE_ACTION]
                    if record.expected_action == RESOLVE_ACTION
                    else ACTION_TO_INDEX[CLARIFY_ACTION]
                )
                multi_action += int(multi_action_index == expected_multi_action)
            decision = realize_unified_decision(record, mode, option)
            expected_option = (
                record.selected_candidate_index
                if record.expected_action == RESOLVE_ACTION
                else NO_SUPPORT_OPTION_INDEX
            )
            if record.expected_action != GENERATE_ACTION:
                structured += int(
                    decision.action == record.expected_action
                    and option == expected_option
                )
            else:
                structured += int(decision.action == GENERATE_ACTION)
            if record.expected_action == RESOLVE_ACTION:
                resolve_total += 1
                exact += int(decision.selected_value == record.expected_value)
                real_candidate += int(real_option == record.selected_candidate_index)
            elif record.expected_action == CLARIFY_ACTION:
                clarify_total += 1
                sentinel += int(option == NO_SUPPORT_OPTION_INDEX)
    return {
        "structured": structured / len(records),
        "exact": exact,
        "resolve_total": resolve_total,
        "real_candidate": real_candidate,
        "sentinel": sentinel,
        "clarify_total": clarify_total,
        "multi_action": multi_action,
        "multi_action_total": multi_action_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    seed_everything(1337)
    device = torch.device(resolve_device(args.device))
    train_pairs = tuple(
        (value, TRAIN_VALUES[(index + 1) % len(TRAIN_VALUES)])
        for index, value in enumerate(TRAIN_VALUES)
    )
    train_records = _records(train_pairs, "train")
    held_records = _records(HELD_OUT_PAIRS, "val", held_out=True)
    tokenizer = ByteBPETokenizer.train(
        "\n".join(record.prompt for record in train_records),
        vocab_size=288,
        min_frequency=2,
    ).with_special_tokens(EXPANDED_CONTROL_TOKENS)
    block_size = 512
    train_examples = encode_unified_records(train_records, tokenizer, block_size)
    held_examples = encode_unified_records(held_records, tokenizer, block_size)
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
    model = UnifiedTypedSpanResolver(backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3)
    initial = None
    final = None
    model.train()
    for step in range(args.steps + 1):
        indices = torch.randint(0, len(train_examples), (8,)).tolist()
        batch = unified_batch(train_examples, indices, device)
        output = model(*batch[:5], batch[5], batch[6], batch[7])
        assert output.loss is not None
        if initial is None:
            initial = float(output.loss.detach())
        final = float(output.loss.detach())
        if step % 50 == 0:
            print(f"step {step:3d}: loss {final:.6f}")
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    train = _assess(model, train_examples, train_records, device)
    held = _assess(model, held_examples, held_records, device)
    print(f"loss: {initial:.6f} -> {final:.6f}")
    print(f"train structured accuracy: {train['structured']:.3f}")
    print(
        f"held-out exact resolution: {held['exact']}/{held['resolve_total']}"
    )
    print(
        f"held-out real-candidate top-1: "
        f"{held['real_candidate']}/{held['resolve_total']}"
    )
    print(
        f"held-out no-support sentinel: "
        f"{held['sentinel']}/{held['clarify_total']}"
    )
    print(
        f"held-out multi-candidate action: "
        f"{held['multi_action']}/{held['multi_action_total']}"
    )
    if train["structured"] != 1.0:
        raise SystemExit("primitive failed to fit the structured training set")
    if held["exact"] != held["resolve_total"]:
        raise SystemExit("primitive failed held-out exact resolution")
    if held["real_candidate"] != held["resolve_total"]:
        raise SystemExit("primitive regressed real-candidate ranking")
    if held["sentinel"] != held["clarify_total"]:
        raise SystemExit("primitive failed held-out no-support routing")
    if held["multi_action"] != held["multi_action_total"]:
        raise SystemExit("primitive failed held-out ambiguity routing")


if __name__ == "__main__":
    main()
