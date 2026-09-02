"""Primitive proof that the Phase 34 BIO heads can learn exact byte spans."""

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
    ExpandedCandidate,
    ExpandedResolverRecord,
)
from story_model.explicit_offset_candidate_proposer import (
    ExplicitOffsetCandidateProposer,
    decode_proposed_spans,
    encode_proposal_records,
    proposal_batch,
    proposal_metric_row,
    proposal_metrics,
)
from story_model.models import build_model
from story_model.runtime import resolve_device, seed_everything
from story_model.unified_typed_span_resolver import UnifiedTypedSpanResolver


TRAIN_VALUES = (
    "alder alcove",
    "birch basket",
    "cinder cabinet",
    "dusk drawer",
    "ember locker",
    "fern niche",
    "flint shelf",
    "frost trunk",
)
HELD_OUT_VALUES = (
    "juniper case",
    "moss table",
    "quartz bureau",
    "raven armoire",
)


def _records(values: tuple[str, ...], split: str):
    records = []
    for index, value in enumerate(values):
        alternative = values[(index + 1) % len(values)]
        identity = f"phase34_primitive_{split}_{index:03d}"
        candidates = (
            ExpandedCandidate(value, "container"),
            ExpandedCandidate(alternative, "container"),
        )
        records.append(
            ExpandedResolverRecord(
                record_id=f"{identity}:resolve",
                source_context_id=identity,
                conversation_id=identity,
                split=split,
                skill="multi_turn_memory",
                case="supported",
                prompt=(
                    f"Verified container: {value}. "
                    "Question: where is the token?"
                ),
                expected_action=RESOLVE_ACTION,
                expected_type="container",
                candidates=candidates,
                selected_candidate_index=0,
                response_template=(
                    "The recorded location is <|resolved_value|>."
                ),
                expected_value=value,
                alternative_value=alternative,
                source_phase="phase31",
            )
        )
        records.append(
            ExpandedResolverRecord(
                record_id=f"{identity}:clarify",
                source_context_id=identity,
                conversation_id=f"{identity}:clarify",
                split=split,
                skill="multi_turn_memory",
                case="missing_evidence",
                prompt="No container was recorded. Where is the token?",
                expected_action=CLARIFY_ACTION,
                expected_type="container",
                candidates=candidates,
                selected_candidate_index=None,
                response_template=CLARIFICATION_RESPONSE,
                expected_value=value,
                alternative_value=alternative,
                source_phase="phase31",
            )
        )
        records.append(
            ExpandedResolverRecord(
                record_id=f"{identity}:generate",
                source_context_id=identity,
                conversation_id=f"{identity}:generate",
                split=split,
                skill="multi_turn_memory",
                case="ordinary_generation",
                prompt="Acknowledge the request without resolving a value.",
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
def _assess(model, examples, records, tokenizer, device):
    model.eval()
    rows = []
    for start in range(0, len(examples), 8):
        indices = tuple(range(start, min(start + 8, len(examples))))
        batch = proposal_batch(
            examples, indices, tokenizer, model.source_width, device
        )
        output = model(batch[0], batch[1], batch[2])
        logits = output.tag_logits.detach().cpu()
        for offset, index in enumerate(indices):
            spans = decode_proposed_spans(
                records[index].prompt, examples[index], logits[offset], tokenizer
            )
            rows.append(
                proposal_metric_row(
                    records[index], examples[index].gold_spans, spans
                )
            )
    return proposal_metrics(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    seed_everything(1337)
    device = torch.device(resolve_device(args.device))
    train_records = _records(TRAIN_VALUES, "train")
    held_records = _records(HELD_OUT_VALUES, "val")
    tokenizer = ByteBPETokenizer.train(
        "\n".join(record.prompt for record in train_records),
        vocab_size=288,
        min_frequency=2,
    ).with_special_tokens(EXPANDED_CONTROL_TOKENS)
    block_size = 256
    train_examples = encode_proposal_records(
        train_records, tokenizer, block_size
    )
    held_examples = encode_proposal_records(held_records, tokenizer, block_size)
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
    model = ExplicitOffsetCandidateProposer(
        UnifiedTypedSpanResolver(backbone), tokenizer
    ).to(device)
    optimizer = torch.optim.AdamW(tuple(model.proposer_parameters()), lr=3.0e-3)
    initial = None
    final = None
    for step in range(args.steps + 1):
        model.train()
        indices = torch.randint(0, len(train_examples), (8,)).tolist()
        batch = proposal_batch(
            train_examples, indices, tokenizer, model.source_width, device
        )
        output = model(*batch)
        assert output.loss is not None
        if initial is None:
            initial = float(output.loss.detach())
        final = float(output.loss.detach())
        if step % 100 == 0:
            print(f"step {step:3d}: loss {final:.6f}")
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(tuple(model.proposer_parameters()), 1.0)
        optimizer.step()
    train = _assess(model, train_examples, train_records, tokenizer, device)
    held = _assess(model, held_examples, held_records, tokenizer, device)
    print(f"loss: {initial:.6f} -> {final:.6f}")
    print(
        "train exact span F1 / answer recall: "
        f"{train['exact_span_f1']:.3f} / "
        f"{train['answer_candidate_recall']:.3f}"
    )
    print(
        "held-out exact span F1 / answer recall: "
        f"{held['exact_span_f1']:.3f} / "
        f"{held['answer_candidate_recall']:.3f}"
    )
    if (
        train["exact_span_f1"] < 1.0
        or train["answer_candidate_recall"] < 1.0
        or train["offset_validity_rate"] < 1.0
    ):
        raise SystemExit("explicit-offset proposer primitive failed")
    print("explicit-offset proposer primitive: passed")


if __name__ == "__main__":
    main()
