"""Phase 33 count-conditioned routing for typed whole-span resolution.

Phase 32 proved that the candidate scorer ranks every real candidate correctly,
but its independent action head can still reject that result. Phase 33 keeps
the exact Phase 32 parameterization and uses evidence eligibility to decide
which part of the structured decision is deterministic. Zero eligible real
candidates clarifies, one resolves directly, and two or more defer the
resolve-versus-clarify decision to the legacy action head. Raw candidate
ranking remains a separately supervised objective on every resolve row.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from story_model.data import ByteBPETokenizer
from story_model.expanded_typed_span_resolver import (
    ACTION_TO_INDEX,
    CANDIDATE_MARKER,
    CLARIFY_ACTION,
    DECISION_MARKER,
    EXPANDED_CONTROL_TOKENS,
    GENERATE_ACTION,
    MAX_CANDIDATES,
    RESOLVER_ACTIONS,
    RESOLVER_MARKER,
    RESOLVE_ACTION,
    ExpandedResolverDecision,
    ExpandedResolverRecord,
    encode_expanded_record,
    realize_expanded_decision,
    summarize_expanded_predictions,
)


STRUCTURED_MODE = "structured"
GENERATE_MODE = "generate"
ROUTING_MODES = (STRUCTURED_MODE, GENERATE_MODE)
MODE_TO_INDEX = {mode: index for index, mode in enumerate(ROUTING_MODES)}
NO_SUPPORT_OPTION_INDEX = MAX_CANDIDATES
MAX_STRUCTURED_OPTIONS = MAX_CANDIDATES + 1
SUPPORT_MASK_VERSION = 2


@dataclass(frozen=True)
class EncodedUnifiedExample:
    record_id: str
    input_ids: tuple[int, ...]
    sequence_tokens: int
    option_view_input_ids: tuple[tuple[int, ...], ...]
    option_view_sequence_tokens: tuple[int, ...]
    option_valid_mask: tuple[bool, ...]
    mode_target: int
    option_target: int
    real_candidate_count: int


def _padded(tokens: list[int], block_size: int, label: str) -> tuple[int, ...]:
    if len(tokens) > block_size:
        raise ValueError(
            f"{label} needs {len(tokens)} tokens but block_size is {block_size}"
        )
    return tuple(tokens + [0] * (block_size - len(tokens)))


def _contains_exact_value(prompt: str, value: str) -> bool:
    """Return whether ``value`` occurs on lexical boundaries in ``prompt``."""

    offset = 0
    while True:
        start = prompt.find(value, offset)
        if start < 0:
            return False
        end = start + len(value)
        left_boundary = (
            start == 0
            or not value[0].isalnum()
            or not prompt[start - 1].isalnum()
        )
        right_boundary = (
            end == len(prompt)
            or not value[-1].isalnum()
            or not prompt[end].isalnum()
        )
        if left_boundary and right_boundary:
            return True
        offset = start + 1


def candidate_is_evidence_eligible(
    record: ExpandedResolverRecord, index: int
) -> bool:
    """Check whether one typed candidate is addressable in the evidence.

    Eligibility means a type-compatible lexical mention. It deliberately does
    not claim that the candidate is asserted true: relational tasks such as
    ``scene_route`` can mention multiple candidates and require a separate
    resolve-versus-clarify decision.
    """

    if not 0 <= index < len(record.candidates):
        return False
    candidate = record.candidates[index]
    return (
        record.expected_type is not None
        and candidate.value_type == record.expected_type
        and _contains_exact_value(record.prompt, candidate.text)
    )


def _no_support_prompt(record: ExpandedResolverRecord) -> str:
    """Build the legacy sentinel view retained by the unified input shape.

    Real candidate views replace one supplied value with ``CANDIDATE_MARKER``.
    The no-support view applies the same operation to every supplied candidate.
    Its suffix deliberately has the same compact shape as a real candidate
    view so the fifth option does not add sequence pressure. Phase 33c routes
    this sentinel from candidate count and the ambiguity action head rather
    than treating its raw candidate score as evidence truth.
    """

    prompt = record.prompt
    for candidate in sorted(
        record.candidates, key=lambda item: len(item.text), reverse=True
    ):
        prompt = prompt.replace(candidate.text, CANDIDATE_MARKER)
    return (
        f"{prompt}{RESOLVER_MARKER}\n"
        f"expected_type: {record.expected_type or 'none'}\n"
        "candidate_type: none\n"
        f"{DECISION_MARKER}\n"
    )


def encode_unified_record(
    record: ExpandedResolverRecord,
    tokenizer: ByteBPETokenizer,
    block_size: int,
) -> EncodedUnifiedExample:
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise TypeError("unified typed-span resolution requires byte-BPE")
    missing = set(EXPANDED_CONTROL_TOKENS) - set(tokenizer.special_token_ids)
    if missing:
        raise ValueError(
            "unified resolver tokenizer is missing control tokens: "
            + ", ".join(sorted(missing))
        )
    if len(record.candidates) > MAX_CANDIDATES:
        raise ValueError("record exceeds Phase 33 real-candidate capacity")

    # Reuse the Phase 32 encoding byte-for-byte.  This is important: changing
    # BPE call boundaries can change token IDs even when the visible text is
    # identical, which would confound the routing experiment.
    phase32 = encode_expanded_record(record, tokenizer, block_size)
    option_views = list(phase32.candidate_view_input_ids)
    option_lengths = list(phase32.candidate_view_sequence_tokens)
    option_valid = [
        candidate_is_evidence_eligible(record, position)
        for position in range(MAX_CANDIDATES)
    ]

    no_support_tokens = tokenizer.encode(_no_support_prompt(record))
    option_views.append(
        _padded(
            no_support_tokens,
            block_size,
            f"no-support view for {record.record_id}",
        )
    )
    option_lengths.append(len(no_support_tokens))
    # The sentinel must remain representable on every structured row. A
    # relational task can mention multiple candidates without supplying the
    # fact that distinguishes them, so lexical candidate presence cannot make
    # clarification unavailable.
    option_valid.append(record.expected_type is not None)

    mode_target = MODE_TO_INDEX[
        GENERATE_MODE
        if record.expected_action == GENERATE_ACTION
        else STRUCTURED_MODE
    ]
    if record.expected_action == RESOLVE_ACTION:
        if record.selected_candidate_index is None:
            raise ValueError("resolve row has no selected candidate")
        option_target = record.selected_candidate_index
    elif record.expected_action == CLARIFY_ACTION:
        option_target = NO_SUPPORT_OPTION_INDEX
    else:
        option_target = -100
    if option_target != -100 and not option_valid[option_target]:
        raise ValueError(
            f"{record.record_id} target option is not evidence-eligible"
        )

    return EncodedUnifiedExample(
        record_id=record.record_id,
        input_ids=phase32.input_ids,
        sequence_tokens=phase32.sequence_tokens,
        option_view_input_ids=tuple(option_views),
        option_view_sequence_tokens=tuple(option_lengths),
        option_valid_mask=tuple(option_valid),
        mode_target=mode_target,
        option_target=option_target,
        real_candidate_count=len(record.candidates),
    )


def encode_unified_records(
    records: Iterable[ExpandedResolverRecord],
    tokenizer: ByteBPETokenizer,
    block_size: int,
) -> tuple[EncodedUnifiedExample, ...]:
    return tuple(
        encode_unified_record(record, tokenizer, block_size) for record in records
    )


def unified_batch(
    examples: tuple[EncodedUnifiedExample, ...],
    indices: Iterable[int],
    device: Union[str, torch.device],
) -> tuple[torch.Tensor, ...]:
    selected = tuple(examples[index] for index in indices)
    if not selected:
        raise ValueError("unified resolver batch cannot be empty")
    return (
        torch.tensor(
            [example.input_ids for example in selected],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [example.sequence_tokens for example in selected],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [example.option_view_input_ids for example in selected],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [example.option_view_sequence_tokens for example in selected],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [example.option_valid_mask for example in selected],
            dtype=torch.bool,
            device=device,
        ),
        torch.tensor(
            [example.mode_target for example in selected],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [example.option_target for example in selected],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [example.real_candidate_count for example in selected],
            dtype=torch.long,
            device=device,
        ),
    )


@dataclass
class UnifiedResolverOutput:
    mode_logits: torch.Tensor
    option_logits: torch.Tensor
    raw_option_logits: torch.Tensor
    legacy_action_logits: torch.Tensor
    loss: Optional[torch.Tensor]
    mode_loss: Optional[torch.Tensor]
    ambiguity_action_loss: Optional[torch.Tensor]
    raw_candidate_loss: Optional[torch.Tensor]


def count_conditioned_option_logits(
    raw_option_logits: torch.Tensor,
    option_valid_mask: torch.Tensor,
    legacy_action_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the deterministic 0/1/2+ routing table.

    Returns the routed option logits and each row's number of evidence-eligible
    real candidates. No learned parameters are introduced by this operation.
    """

    if raw_option_logits.shape[-1] != MAX_STRUCTURED_OPTIONS:
        raise ValueError(
            f"expected {MAX_STRUCTURED_OPTIONS} raw structured option logits"
        )
    if option_valid_mask.shape != raw_option_logits.shape:
        raise ValueError("option_valid_mask must match raw option logits")
    minimum = torch.finfo(raw_option_logits.dtype).min
    real_valid = option_valid_mask[:, :MAX_CANDIDATES]
    eligible_counts = real_valid.sum(dim=-1)
    masked_real_logits = raw_option_logits[:, :MAX_CANDIDATES].masked_fill(
        ~real_valid, minimum
    )
    best_real_logits = masked_real_logits.max(dim=-1).values
    best_real_logits = torch.where(
        eligible_counts > 0,
        best_real_logits,
        torch.zeros_like(best_real_logits),
    )
    relative_real_logits = (
        raw_option_logits[:, :MAX_CANDIDATES]
        - best_real_logits.unsqueeze(-1)
    )
    multi_eligible = eligible_counts >= 2
    resolve_margin = (
        legacy_action_logits[:, ACTION_TO_INDEX[RESOLVE_ACTION]]
        - legacy_action_logits[:, ACTION_TO_INDEX[CLARIFY_ACTION]]
    )
    relative_real_logits = relative_real_logits + torch.where(
        multi_eligible,
        resolve_margin,
        torch.zeros_like(resolve_margin),
    ).unsqueeze(-1)
    routed_real_logits = relative_real_logits.masked_fill(~real_valid, minimum)

    sentinel_input_valid = option_valid_mask[:, NO_SUPPORT_OPTION_INDEX]
    sentinel_runtime_valid = sentinel_input_valid & (eligible_counts != 1)
    sentinel_logits = torch.zeros_like(resolve_margin).masked_fill(
        ~sentinel_runtime_valid, minimum
    )
    return (
        torch.cat((routed_real_logits, sentinel_logits.unsqueeze(-1)), dim=-1),
        eligible_counts,
    )


class UnifiedTypedSpanResolver(nn.Module):
    """Phase 32-compatible parameters with count-conditioned routing."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        required = (
            "token_embeddings",
            "position_embeddings",
            "blocks",
            "final_norm",
            "block_size",
        )
        if not all(hasattr(backbone, attribute) for attribute in required):
            raise TypeError("resolver backbone must be a Transformer model")
        self.backbone = backbone
        embedding_dim = int(backbone.token_embeddings.embedding_dim)
        # These names and shapes intentionally match the Phase 32 checkpoint.
        self.action_head = nn.Linear(embedding_dim, len(RESOLVER_ACTIONS))
        self.candidate_score = nn.Linear(embedding_dim, 1)
        nn.init.normal_(self.action_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.action_head.bias)
        nn.init.normal_(self.candidate_score.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.candidate_score.bias)

    @property
    def block_size(self) -> int:
        return int(self.backbone.block_size)

    def hidden_states(self, tokens: torch.Tensor) -> torch.Tensor:
        _, sequence = tokens.shape
        if sequence > self.block_size:
            raise ValueError(
                f"Sequence length {sequence} exceeds block size {self.block_size}"
            )
        hidden = self.backbone.token_embeddings(tokens)
        if self.backbone.position_embeddings is not None:
            positions = torch.arange(sequence, device=tokens.device)
            hidden = hidden + self.backbone.position_embeddings(positions)
        for block in self.backbone.blocks:
            hidden = block(hidden)
        return self.backbone.final_norm(hidden)

    def forward(
        self,
        tokens: torch.Tensor,
        sequence_lengths: torch.Tensor,
        option_view_tokens: torch.Tensor,
        option_view_sequence_lengths: torch.Tensor,
        option_valid_mask: torch.Tensor,
        mode_targets: Optional[torch.Tensor] = None,
        option_targets: Optional[torch.Tensor] = None,
        real_candidate_counts: Optional[torch.Tensor] = None,
    ) -> UnifiedResolverOutput:
        batch, option_count, sequence = option_view_tokens.shape
        if option_count != MAX_STRUCTURED_OPTIONS:
            raise ValueError(
                f"Phase 33 expects {MAX_STRUCTURED_OPTIONS} structured options"
            )
        combined_tokens = torch.cat(
            (tokens, option_view_tokens.reshape(-1, sequence)), dim=0
        )
        combined_hidden = self.hidden_states(combined_tokens)
        hidden = combined_hidden[:batch]
        option_hidden = combined_hidden[batch:].reshape(
            batch, option_count, sequence, -1
        )

        positions = sequence_lengths - 1
        queries = hidden[torch.arange(batch, device=tokens.device), positions]
        legacy_action_logits = self.action_head(queries)
        structured_logit = torch.logsumexp(
            legacy_action_logits[:, : ACTION_TO_INDEX[GENERATE_ACTION]], dim=-1
        )
        mode_logits = torch.stack(
            (
                structured_logit,
                legacy_action_logits[:, ACTION_TO_INDEX[GENERATE_ACTION]],
            ),
            dim=-1,
        )

        option_positions = option_view_sequence_lengths - 1
        batch_indices = torch.arange(batch, device=tokens.device).unsqueeze(1)
        option_indices = torch.arange(option_count, device=tokens.device).unsqueeze(0)
        option_states = option_hidden[
            batch_indices, option_indices, option_positions
        ]
        raw_option_logits = self.candidate_score(option_states).squeeze(-1)

        # Build the structured decision without asking one softmax to learn
        # three qualitatively different routing cases.
        option_logits, eligible_counts = count_conditioned_option_logits(
            raw_option_logits,
            option_valid_mask,
            legacy_action_logits,
        )
        multi_eligible = eligible_counts >= 2

        mode_loss = None
        ambiguity_action_loss = None
        raw_candidate_loss = None
        loss = None
        if mode_targets is not None:
            mode_loss = F.cross_entropy(mode_logits, mode_targets)
            loss = mode_loss
        if option_targets is not None:
            structured_rows = option_targets != -100
            ambiguity_rows = structured_rows & multi_eligible
            if ambiguity_rows.any():
                ambiguity_action_targets = torch.where(
                    option_targets[ambiguity_rows] == NO_SUPPORT_OPTION_INDEX,
                    torch.full_like(
                        option_targets[ambiguity_rows],
                        ACTION_TO_INDEX[CLARIFY_ACTION],
                    ),
                    torch.full_like(
                        option_targets[ambiguity_rows],
                        ACTION_TO_INDEX[RESOLVE_ACTION],
                    ),
                )
                ambiguity_action_loss = F.cross_entropy(
                    legacy_action_logits[ambiguity_rows, :2],
                    ambiguity_action_targets,
                )
                loss = (
                    ambiguity_action_loss
                    if loss is None
                    else loss + ambiguity_action_loss
                )
            resolve_rows = (
                (option_targets >= 0) & (option_targets < MAX_CANDIDATES)
            )
            if resolve_rows.any():
                if real_candidate_counts is None:
                    raise ValueError(
                        "real_candidate_counts are required for raw ranking loss"
                    )
                positions = torch.arange(
                    MAX_CANDIDATES, device=tokens.device
                ).unsqueeze(0)
                inventory_valid = positions < real_candidate_counts.unsqueeze(1)
                raw_candidate_logits = raw_option_logits[
                    :, :MAX_CANDIDATES
                ].masked_fill(
                    ~inventory_valid,
                    torch.finfo(raw_option_logits.dtype).min,
                )
                raw_candidate_loss = F.cross_entropy(
                    raw_candidate_logits[resolve_rows],
                    option_targets[resolve_rows],
                )
                loss = (
                    raw_candidate_loss
                    if loss is None
                    else loss + raw_candidate_loss
                )
        return UnifiedResolverOutput(
            mode_logits=mode_logits,
            option_logits=option_logits,
            raw_option_logits=raw_option_logits,
            legacy_action_logits=legacy_action_logits,
            loss=loss,
            mode_loss=mode_loss,
            ambiguity_action_loss=ambiguity_action_loss,
            raw_candidate_loss=raw_candidate_loss,
        )


def unified_action(mode_index: int, option_index: int) -> str:
    if mode_index == MODE_TO_INDEX[GENERATE_MODE]:
        return GENERATE_ACTION
    if mode_index != MODE_TO_INDEX[STRUCTURED_MODE]:
        raise ValueError(f"unknown routing mode index: {mode_index}")
    if option_index == NO_SUPPORT_OPTION_INDEX:
        return CLARIFY_ACTION
    return RESOLVE_ACTION


def realize_unified_decision(
    record: ExpandedResolverRecord,
    mode_index: int,
    option_index: int,
) -> ExpandedResolverDecision:
    action = unified_action(mode_index, option_index)
    if action == RESOLVE_ACTION and not candidate_is_evidence_eligible(
        record, option_index
    ):
        decision = realize_expanded_decision(record, CLARIFY_ACTION, None)
        return replace(decision, guarded=True)
    candidate_index = option_index if action == RESOLVE_ACTION else None
    return realize_expanded_decision(record, action, candidate_index)


def _rate(rows: tuple[dict, ...], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows) if rows else 0.0


def _unified_metrics(rows: tuple[dict, ...]) -> dict:
    resolve = tuple(row for row in rows if row["expected_action"] == RESOLVE_ACTION)
    clarify = tuple(row for row in rows if row["expected_action"] == CLARIFY_ACTION)
    multi_eligible = tuple(
        row
        for row in rows
        if row.get("eligible_candidate_count", 0) >= 2
        and row["expected_action"] != GENERATE_ACTION
    )
    return {
        "mode_accuracy": _rate(rows, "mode_correct"),
        "real_candidate_top1_accuracy": _rate(resolve, "real_candidate_correct"),
        "resolve_option_accuracy": _rate(resolve, "option_correct"),
        "clarify_sentinel_accuracy": _rate(clarify, "option_correct"),
        "no_support_false_positive_rate": _rate(resolve, "no_support_selected"),
        "multi_candidate_action_examples": len(multi_eligible),
        "multi_candidate_action_accuracy": _rate(
            multi_eligible, "multi_candidate_action_correct"
        ),
    }


def summarize_unified_predictions(rows: Iterable[dict]) -> dict:
    rows = tuple(rows)
    summary = summarize_expanded_predictions(rows)
    summary.update(_unified_metrics(rows))

    by_skill = {}
    for skill in sorted({row["skill"] for row in rows}):
        group = tuple(row for row in rows if row["skill"] == skill)
        by_skill[skill] = _unified_metrics(group)
    for skill, metrics in by_skill.items():
        summary["per_skill"][skill].update(metrics)

    resolve = tuple(row for row in rows if row["expected_action"] == RESOLVE_ACTION)
    for width in sorted({str(row["candidate_count"]) for row in resolve}):
        group = tuple(row for row in resolve if str(row["candidate_count"]) == width)
        summary["per_candidate_width"][width].update(_unified_metrics(group))
    for skill, width_groups in summary["per_skill_candidate_width"].items():
        for width, metrics in width_groups.items():
            group = tuple(
                row
                for row in resolve
                if row["skill"] == skill and str(row["candidate_count"]) == width
            )
            metrics.update(_unified_metrics(group))
    return summary
