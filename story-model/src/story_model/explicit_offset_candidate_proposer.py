"""Phase 34 typed candidate proposal with explicit UTF-8 byte offsets.

The Phase 33c resolver remains frozen.  A small byte-level BIO tagger learns
to propose typed evidence spans directly from the serialized prompt.  Exact
offsets, rather than substring search at resolver time, are the contract
between proposal and deterministic whole-span realization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from story_model.data import ByteBPETokenizer
from story_model.expanded_typed_span_resolver import (
    CANDIDATE_MARKER,
    CLARIFICATION_RESPONSE,
    CLARIFY_ACTION,
    DECISION_MARKER,
    EXPANDED_CONTROL_TOKENS,
    GENERATE_ACTION,
    MAX_CANDIDATES,
    RESOLVER_MARKER,
    RESOLVE_ACTION,
    SKILL_RESPONSE_FRAMES,
    SKILL_VALUE_TYPES,
    ExpandedCandidate,
    ExpandedResolverDecision,
    ExpandedResolverRecord,
)
from story_model.unified_typed_span_resolver import (
    GENERATE_MODE,
    MODE_TO_INDEX,
    NO_SUPPORT_OPTION_INDEX,
    UnifiedTypedSpanResolver,
)


EXPLICIT_OFFSET_PROPOSER_VERSION = 1
BOUNDARY_OBJECTIVE_VERSION = 1
TOKEN_WIDTH_GEOMETRY_VERSION = 1
TOKEN_END_GEOMETRY_VERSION = 1
PROPOSAL_TYPES = tuple(sorted(set(SKILL_VALUE_TYPES.values())))
TYPE_TO_INDEX = {value_type: index for index, value_type in enumerate(PROPOSAL_TYPES)}
OUTSIDE_TAG = 0
IGNORE_TAG = -100
MAX_DECODED_SPANS = 16
EXCLUDED_PROPOSER_CASES = ("wrong_type",)
PERMISSIVE_DECODE_POLICY = "permissive"
STRICT_DECODE_POLICY = "strict"
DECODE_POLICIES = (PERMISSIVE_DECODE_POLICY, STRICT_DECODE_POLICY)


def checkpoint_uses_token_width_geometry(extra: dict) -> bool:
    """Return whether proposer metadata selects the Phase 34f geometry."""

    version = extra.get("token_width_geometry_version")
    if version is None:
        return False
    if type(version) is not int or (
        version != TOKEN_WIDTH_GEOMETRY_VERSION
    ):
        raise ValueError(
            "checkpoint has an unsupported token-width geometry version"
        )
    return True


def checkpoint_uses_token_end_geometry(extra: dict) -> bool:
    """Return whether proposer metadata selects the Phase 34g geometry."""

    version = extra.get("token_end_geometry_version")
    if version is None:
        return False
    if type(version) is not int or version != TOKEN_END_GEOMETRY_VERSION:
        raise ValueError(
            "checkpoint has an unsupported token-end geometry version"
        )
    return True


def begin_tag(value_type: str) -> int:
    if value_type not in TYPE_TO_INDEX:
        raise ValueError(f"unknown proposal type: {value_type}")
    return 1 + 2 * TYPE_TO_INDEX[value_type]


def inside_tag(value_type: str) -> int:
    return begin_tag(value_type) + 1


def tag_type(tag: int) -> Optional[str]:
    if tag == OUTSIDE_TAG:
        return None
    index = (tag - 1) // 2
    if not 0 <= index < len(PROPOSAL_TYPES):
        raise ValueError(f"unknown proposal tag: {tag}")
    return PROPOSAL_TYPES[index]


def tag_is_begin(tag: int) -> bool:
    return tag != OUTSIDE_TAG and (tag - 1) % 2 == 0


@dataclass(frozen=True)
class EvidenceSpan:
    byte_start: int
    byte_end: int
    text: str
    value_type: str
    score: float = field(default=1.0, compare=False)

    def __post_init__(self) -> None:
        if not 0 <= self.byte_start < self.byte_end:
            raise ValueError("evidence span byte offsets are invalid")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("evidence span text cannot be empty")
        if self.value_type not in TYPE_TO_INDEX:
            raise ValueError(f"unknown evidence span type: {self.value_type}")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("evidence span score must be between zero and one")

    def validate_prompt(self, prompt: str) -> None:
        prompt_bytes = prompt.encode("utf-8")
        if self.byte_end > len(prompt_bytes):
            raise ValueError("evidence span extends beyond the prompt")
        try:
            actual = prompt_bytes[self.byte_start : self.byte_end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("evidence span splits a UTF-8 character") from error
        if actual != self.text:
            raise ValueError(
                f"evidence span text mismatch: expected {self.text!r}, got {actual!r}"
            )

    def to_dict(self) -> dict:
        return {
            "byte_start": self.byte_start,
            "byte_end": self.byte_end,
            "text": self.text,
            "value_type": self.value_type,
            "score": self.score,
        }


def proposal_record_is_eligible(record: ExpandedResolverRecord) -> bool:
    """Return whether a row has a meaningful prompt-only proposal label."""

    return record.case not in EXCLUDED_PROPOSER_CASES


def _exact_occurrences(prompt: str, value: str) -> tuple[tuple[int, int], ...]:
    offsets = []
    cursor = 0
    while True:
        start = prompt.find(value, cursor)
        if start < 0:
            break
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
            offsets.append((start, end))
        cursor = start + 1
    return tuple(offsets)


def _byte_offset(text: str, character_offset: int) -> int:
    return len(text[:character_offset].encode("utf-8"))


def gold_evidence_spans(record: ExpandedResolverRecord) -> tuple[EvidenceSpan, ...]:
    """Derive explicit occurrence annotations from the current gold inventory."""

    if not proposal_record_is_eligible(record):
        raise ValueError(
            f"{record.record_id} case {record.case!r} is not a proposer example"
        )
    if record.expected_type is None:
        return ()

    spans = []
    for candidate in record.candidates:
        if candidate.value_type != record.expected_type:
            continue
        for character_start, character_end in _exact_occurrences(
            record.prompt, candidate.text
        ):
            span = EvidenceSpan(
                byte_start=_byte_offset(record.prompt, character_start),
                byte_end=_byte_offset(record.prompt, character_end),
                text=candidate.text,
                value_type=candidate.value_type,
            )
            span.validate_prompt(record.prompt)
            spans.append(span)

    spans = sorted(
        set(spans),
        key=lambda span: (span.byte_start, span.byte_end, span.value_type),
    )
    for previous, current in zip(spans, spans[1:]):
        if current.byte_start < previous.byte_end:
            raise ValueError(f"overlapping gold spans in {record.record_id}")
    return tuple(spans)


def proposal_source_width(tokenizer: ByteBPETokenizer) -> int:
    return max(
        tokenizer.token_byte_length(token_id)
        for token_id in range(tokenizer.vocab_size)
    )


def _proposal_suffix(record: ExpandedResolverRecord) -> str:
    return (
        f"{RESOLVER_MARKER}\n"
        f"expected_type: {record.expected_type or 'none'}\n"
        f"{CANDIDATE_MARKER}\n"
        "propose typed evidence spans\n"
        f"{DECISION_MARKER}\n"
    )


@dataclass(frozen=True)
class EncodedProposalExample:
    record_id: str
    input_ids: tuple[int, ...]
    sequence_tokens: int
    prompt_token_count: int
    prompt_byte_tags: tuple[int, ...]
    gold_spans: tuple[EvidenceSpan, ...]


def encode_proposal_record(
    record: ExpandedResolverRecord,
    tokenizer: ByteBPETokenizer,
    block_size: int,
) -> EncodedProposalExample:
    if not proposal_record_is_eligible(record):
        raise ValueError(f"cannot encode excluded proposer case: {record.case}")
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise TypeError("explicit-offset proposal requires byte-BPE")
    missing = set(EXPANDED_CONTROL_TOKENS) - set(tokenizer.special_token_ids)
    if missing:
        raise ValueError(
            "candidate proposer tokenizer is missing control tokens: "
            + ", ".join(sorted(missing))
        )

    prompt_ids = tokenizer.encode(record.prompt)
    tokens = prompt_ids + tokenizer.encode(_proposal_suffix(record))
    if len(tokens) > block_size:
        raise ValueError(
            f"proposal input for {record.record_id} needs {len(tokens)} tokens "
            f"but block_size is {block_size}"
        )
    spans = gold_evidence_spans(record)
    prompt_bytes = record.prompt.encode("utf-8")
    tags = [OUTSIDE_TAG] * len(prompt_bytes)
    for span in spans:
        tags[span.byte_start] = begin_tag(span.value_type)
        for position in range(span.byte_start + 1, span.byte_end):
            tags[position] = inside_tag(span.value_type)

    reconstructed = b"".join(tokenizer.token_bytes(token) for token in prompt_ids)
    if reconstructed != prompt_bytes:
        raise RuntimeError("prompt token bytes do not reproduce the prompt")
    return EncodedProposalExample(
        record_id=record.record_id,
        input_ids=tuple(tokens + [0] * (block_size - len(tokens))),
        sequence_tokens=len(tokens),
        prompt_token_count=len(prompt_ids),
        prompt_byte_tags=tuple(tags),
        gold_spans=spans,
    )


def encode_proposal_records(
    records: Iterable[ExpandedResolverRecord],
    tokenizer: ByteBPETokenizer,
    block_size: int,
) -> tuple[EncodedProposalExample, ...]:
    return tuple(
        encode_proposal_record(record, tokenizer, block_size)
        for record in records
        if proposal_record_is_eligible(record)
    )


def proposal_batch(
    examples: tuple[EncodedProposalExample, ...],
    indices: Iterable[int],
    tokenizer: ByteBPETokenizer,
    source_width: int,
    device: Union[str, torch.device],
) -> tuple[torch.Tensor, ...]:
    selected = tuple(examples[index] for index in indices)
    if not selected:
        raise ValueError("candidate proposer batch cannot be empty")
    block_size = len(selected[0].input_ids)
    targets = torch.full(
        (len(selected), block_size, source_width),
        IGNORE_TAG,
        dtype=torch.long,
        device=device,
    )
    for batch_index, example in enumerate(selected):
        byte_cursor = 0
        for token_position in range(example.prompt_token_count):
            token_id = example.input_ids[token_position]
            width = tokenizer.token_byte_length(token_id)
            if width > source_width:
                raise RuntimeError("prompt token exceeds proposer source width")
            token_tags = example.prompt_byte_tags[byte_cursor : byte_cursor + width]
            if len(token_tags) != width:
                raise RuntimeError("prompt byte labels do not cover token bytes")
            targets[batch_index, token_position, :width] = torch.tensor(
                token_tags, dtype=torch.long, device=device
            )
            byte_cursor += width
        if byte_cursor != len(example.prompt_byte_tags):
            raise RuntimeError("prompt byte labels were not consumed exactly")
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
            [example.prompt_token_count for example in selected],
            dtype=torch.long,
            device=device,
        ),
        targets,
    )


@dataclass
class CandidateProposerOutput:
    tag_logits: torch.Tensor
    loss: Optional[torch.Tensor]
    tag_loss: Optional[torch.Tensor] = None
    boundary_start_loss: Optional[torch.Tensor] = None
    boundary_end_loss: Optional[torch.Tensor] = None
    boundary_loss: Optional[torch.Tensor] = None
    boundary_start_positions: int = 0
    boundary_end_positions: int = 0


@dataclass
class BoundaryAuxiliaryLosses:
    start_loss: torch.Tensor
    end_loss: torch.Tensor
    combined_loss: torch.Tensor
    start_positions: int
    end_positions: int


def boundary_auxiliary_losses(
    tag_logits: torch.Tensor,
    tag_targets: torch.Tensor,
) -> BoundaryAuxiliaryLosses:
    """Compute unweighted CE at gold span starts and immediate ends."""

    if tag_targets.shape != tag_logits.shape[:-1]:
        raise ValueError("tag targets must match byte-logit positions")
    start_logits = []
    start_targets = []
    end_logits = []
    end_targets = []
    for batch_index in range(len(tag_targets)):
        supervised = tag_targets[batch_index] != IGNORE_TAG
        sequence_targets = tag_targets[batch_index][supervised]
        sequence_logits = tag_logits[batch_index][supervised]
        if not len(sequence_targets):
            continue
        starts = (sequence_targets > OUTSIDE_TAG) & (
            (sequence_targets - 1).remainder(2) == 0
        )
        if bool(starts.any()):
            start_logits.append(sequence_logits[starts])
            start_targets.append(sequence_targets[starts])
        if len(sequence_targets) > 1:
            ends = (
                (sequence_targets[:-1] > OUTSIDE_TAG)
                & (sequence_targets[1:] == OUTSIDE_TAG)
            )
            if bool(ends.any()):
                end_logits.append(sequence_logits[1:][ends])
                end_targets.append(sequence_targets[1:][ends])

    zero = tag_logits[..., 0].reshape(-1)[0] * 0.0
    start_positions = sum(len(targets) for targets in start_targets)
    end_positions = sum(len(targets) for targets in end_targets)
    start_loss = (
        F.cross_entropy(torch.cat(start_logits), torch.cat(start_targets))
        if start_positions
        else zero
    )
    end_loss = (
        F.cross_entropy(torch.cat(end_logits), torch.cat(end_targets))
        if end_positions
        else zero
    )
    components = []
    if start_positions:
        components.append(start_loss)
    if end_positions:
        components.append(end_loss)
    combined_loss = (
        torch.stack(components).mean() if components else zero
    )
    return BoundaryAuxiliaryLosses(
        start_loss=start_loss,
        end_loss=end_loss,
        combined_loss=combined_loss,
        start_positions=start_positions,
        end_positions=end_positions,
    )


@dataclass(frozen=True)
class ProposalDecodeResult:
    spans: tuple[EvidenceSpan, ...]
    orphan_inside_tags: int
    mismatched_inside_tags: int
    invalid_utf8_spans: int
    truncated_spans: int


class ExplicitOffsetCandidateProposer(nn.Module):
    """Frozen Phase 33c resolver plus a trainable byte-level BIO tagger."""

    def __init__(
        self,
        resolver: UnifiedTypedSpanResolver,
        tokenizer: ByteBPETokenizer,
        token_width_geometry: bool = False,
        token_end_geometry: bool = False,
    ) -> None:
        super().__init__()
        if token_width_geometry and token_end_geometry:
            raise ValueError(
                "token-width and token-end geometry are mutually exclusive"
            )
        self.resolver = resolver
        for parameter in self.resolver.parameters():
            parameter.requires_grad_(False)
        embedding_dim = int(resolver.backbone.token_embeddings.embedding_dim)
        self.source_width = proposal_source_width(tokenizer)
        byte_ids = torch.zeros(
            tokenizer.vocab_size, self.source_width, dtype=torch.long
        )
        byte_mask = torch.zeros(
            tokenizer.vocab_size, self.source_width, dtype=torch.bool
        )
        for token_id in range(tokenizer.vocab_size):
            token_bytes = tokenizer.token_bytes(token_id)
            byte_ids[token_id, : len(token_bytes)] = torch.tensor(
                list(token_bytes), dtype=torch.long
            )
            byte_mask[token_id, : len(token_bytes)] = True
        self.register_buffer("token_byte_ids", byte_ids, persistent=False)
        self.register_buffer("token_byte_mask", byte_mask, persistent=False)
        self.proposal_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.proposal_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.proposal_offset_embedding = nn.Embedding(
            self.source_width, embedding_dim
        )
        self.proposal_tag = nn.Linear(
            embedding_dim, 1 + 2 * len(PROPOSAL_TYPES)
        )
        self.token_width_geometry = bool(token_width_geometry)
        self.proposal_token_width_embedding = None
        if self.token_width_geometry:
            # Preserve the Phase 34d RNG trajectory and initial logits. The
            # zero rows learn independently once examples of each width arrive.
            with torch.random.fork_rng(devices=[]):
                self.proposal_token_width_embedding = nn.Embedding(
                    self.source_width + 1, embedding_dim
                )
            nn.init.zeros_(self.proposal_token_width_embedding.weight)
        self.token_end_geometry = bool(token_end_geometry)
        self.proposal_token_end_embedding = None
        if self.token_end_geometry:
            with torch.random.fork_rng(devices=[]):
                self.proposal_token_end_embedding = nn.Embedding(
                    2, embedding_dim, padding_idx=0
                )
            nn.init.zeros_(self.proposal_token_end_embedding.weight)
        self.register_buffer(
            "tag_class_weights",
            torch.tensor(
                [0.05]
                + [value for _ in PROPOSAL_TYPES for value in (1.0, 0.5)],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        for module in (
            self.proposal_query,
            self.proposal_key,
            self.proposal_offset_embedding,
            self.proposal_tag,
        ):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        self.resolver.eval()
        return self

    def proposer_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("resolver."):
                yield parameter

    def forward(
        self,
        tokens: torch.Tensor,
        sequence_lengths: torch.Tensor,
        prompt_token_counts: torch.Tensor,
        tag_targets: Optional[torch.Tensor] = None,
        boundary_loss_weight: float = 0.0,
    ) -> CandidateProposerOutput:
        if boundary_loss_weight < 0.0:
            raise ValueError("boundary loss weight cannot be negative")
        del prompt_token_counts
        self.resolver.eval()
        with torch.no_grad():
            hidden = self.resolver.hidden_states(tokens)
            byte_embeddings = self.resolver.backbone.token_embeddings(
                self.token_byte_ids[tokens]
            )
        positions = sequence_lengths - 1
        queries = hidden[torch.arange(len(tokens), device=tokens.device), positions]
        base = self.proposal_key(hidden) + self.proposal_query(queries).unsqueeze(1)
        offsets = torch.arange(self.source_width, device=tokens.device)
        byte_inputs = (
            base.unsqueeze(2)
            + byte_embeddings
            + self.proposal_offset_embedding(offsets)[None, None, :, :]
        )
        if self.proposal_token_width_embedding is not None:
            token_widths = self.token_byte_mask[tokens].sum(dim=-1)
            width_states = self.proposal_token_width_embedding(token_widths)
            byte_inputs = byte_inputs + width_states.unsqueeze(2)
        if self.proposal_token_end_embedding is not None:
            token_widths = self.token_byte_mask[tokens].sum(dim=-1)
            token_ends = offsets[None, None, :] == (
                token_widths.unsqueeze(-1) - 1
            )
            byte_inputs = byte_inputs + self.proposal_token_end_embedding(
                token_ends.long()
            )
        byte_states = torch.tanh(byte_inputs)
        tag_logits = self.proposal_tag(byte_states)
        tag_logits = tag_logits.masked_fill(
            ~self.token_byte_mask[tokens].unsqueeze(-1),
            torch.finfo(tag_logits.dtype).min,
        )
        loss = None
        tag_loss = None
        boundary_start_loss = None
        boundary_end_loss = None
        boundary_loss = None
        boundary_start_positions = 0
        boundary_end_positions = 0
        if tag_targets is not None:
            if tag_targets.shape != tag_logits.shape[:-1]:
                raise ValueError("tag targets must match byte-logit positions")
            tag_loss = F.cross_entropy(
                tag_logits.reshape(-1, tag_logits.shape[-1]),
                tag_targets.reshape(-1),
                weight=self.tag_class_weights.to(dtype=tag_logits.dtype),
                ignore_index=IGNORE_TAG,
            )
            boundary = boundary_auxiliary_losses(tag_logits, tag_targets)
            boundary_start_loss = boundary.start_loss
            boundary_end_loss = boundary.end_loss
            boundary_loss = boundary.combined_loss
            boundary_start_positions = boundary.start_positions
            boundary_end_positions = boundary.end_positions
            loss = tag_loss + boundary_loss_weight * boundary_loss
        return CandidateProposerOutput(
            tag_logits=tag_logits,
            loss=loss,
            tag_loss=tag_loss,
            boundary_start_loss=boundary_start_loss,
            boundary_end_loss=boundary_end_loss,
            boundary_loss=boundary_loss,
            boundary_start_positions=boundary_start_positions,
            boundary_end_positions=boundary_end_positions,
        )


def decode_proposed_spans(
    prompt: str,
    example: EncodedProposalExample,
    tag_logits: torch.Tensor,
    tokenizer: ByteBPETokenizer,
    policy: str = PERMISSIVE_DECODE_POLICY,
) -> tuple[EvidenceSpan, ...]:
    """Decode byte BIO tags under one explicit malformed-tag policy."""

    return decode_proposal_result(
        prompt, example, tag_logits, tokenizer, policy=policy
    ).spans


def decode_proposal_result(
    prompt: str,
    example: EncodedProposalExample,
    tag_logits: torch.Tensor,
    tokenizer: ByteBPETokenizer,
    policy: str = PERMISSIVE_DECODE_POLICY,
) -> ProposalDecodeResult:
    """Decode spans and retain diagnostics for malformed BIO transitions."""

    if policy not in DECODE_POLICIES:
        raise ValueError(f"unknown proposal decode policy: {policy}")

    probabilities = F.softmax(tag_logits.float(), dim=-1)
    tags = tag_logits.argmax(dim=-1)
    byte_tags = []
    byte_scores = []
    for token_position in range(example.prompt_token_count):
        token_id = example.input_ids[token_position]
        width = tokenizer.token_byte_length(token_id)
        for byte_offset in range(width):
            tag = int(tags[token_position, byte_offset])
            byte_tags.append(tag)
            byte_scores.append(float(probabilities[token_position, byte_offset, tag]))
    if len(byte_tags) != len(prompt.encode("utf-8")):
        raise RuntimeError("decoded proposal bytes do not cover the prompt")

    spans = []
    active_start = None
    active_type = None
    active_scores = []
    invalid_utf8_spans = 0
    truncated_spans = 0
    orphan_inside_tags = 0
    mismatched_inside_tags = 0
    grammar_active_type = None

    def finish(end: int) -> None:
        nonlocal active_start, active_type, active_scores
        nonlocal invalid_utf8_spans, truncated_spans
        if active_start is None or active_type is None:
            return
        raw = prompt.encode("utf-8")[active_start:end]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            invalid_utf8_spans += 1
            active_start = None
            active_type = None
            active_scores = []
            return
        if text and len(spans) < MAX_DECODED_SPANS:
            span = EvidenceSpan(
                active_start,
                end,
                text,
                active_type,
                sum(active_scores) / len(active_scores),
            )
            span.validate_prompt(prompt)
            spans.append(span)
        elif text:
            truncated_spans += 1
        active_start = None
        active_type = None
        active_scores = []

    for position, (tag, score) in enumerate(zip(byte_tags, byte_scores)):
        value_type = tag_type(tag)
        if value_type is None:
            grammar_active_type = None
            finish(position)
            continue

        is_begin = tag_is_begin(tag)
        if is_begin:
            grammar_active_type = value_type
            finish(position)
            active_start = position
            active_type = value_type
            active_scores = [score]
            continue

        if grammar_active_type is None:
            orphan_inside_tags += 1
        elif value_type != grammar_active_type:
            mismatched_inside_tags += 1
            grammar_active_type = None

        if active_start is not None and value_type == active_type:
            active_scores.append(score)
            continue

        finish(position)
        if policy == PERMISSIVE_DECODE_POLICY:
            active_start = position
            active_type = value_type
            active_scores = [score]
    finish(len(byte_tags))
    return ProposalDecodeResult(
        spans=tuple(spans),
        orphan_inside_tags=orphan_inside_tags,
        mismatched_inside_tags=mismatched_inside_tags,
        invalid_utf8_spans=invalid_utf8_spans,
        truncated_spans=truncated_spans,
    )


def proposed_candidates(
    record: ExpandedResolverRecord,
    spans: Iterable[EvidenceSpan],
) -> tuple[ExpandedCandidate, ...]:
    """Convert offset-backed spans into a bounded unique candidate inventory."""

    if record.expected_type is None:
        return ()
    by_text = {}
    for span in spans:
        span.validate_prompt(record.prompt)
        if span.value_type != record.expected_type:
            continue
        if span.text != span.text.strip() or not span.text or "<|" in span.text:
            continue
        occurrence_bytes = {
            (
                _byte_offset(record.prompt, character_start),
                _byte_offset(record.prompt, character_end),
            )
            for character_start, character_end in _exact_occurrences(
                record.prompt, span.text
            )
        }
        if (span.byte_start, span.byte_end) not in occurrence_bytes:
            continue
        previous = by_text.get(span.text)
        if previous is None or span.score > previous.score:
            by_text[span.text] = span
    ranked = sorted(
        by_text.values(), key=lambda span: (-span.score, span.byte_start, span.text)
    )[:MAX_CANDIDATES]
    ranked.sort(key=lambda span: (span.byte_start, span.byte_end, span.text))
    return tuple(
        ExpandedCandidate(span.text, span.value_type) for span in ranked
    )


def valid_unique_proposed_span_count(
    record: ExpandedResolverRecord,
    spans: Iterable[EvidenceSpan],
) -> int:
    """Count unique runtime-valid values before the four-candidate limit."""

    if record.expected_type is None:
        return 0
    values = set()
    for span in spans:
        span.validate_prompt(record.prompt)
        if span.value_type != record.expected_type:
            continue
        if span.text != span.text.strip() or not span.text or "<|" in span.text:
            continue
        occurrence_bytes = {
            (
                _byte_offset(record.prompt, character_start),
                _byte_offset(record.prompt, character_end),
            )
            for character_start, character_end in _exact_occurrences(
                record.prompt, span.text
            )
        }
        if (span.byte_start, span.byte_end) in occurrence_bytes:
            values.add(span.text)
    return len(values)


def runtime_record_from_spans(
    record: ExpandedResolverRecord,
    spans: Iterable[EvidenceSpan],
) -> ExpandedResolverRecord:
    """Create an unlabeled resolver request without leaking the gold action."""

    if record.expected_action == GENERATE_ACTION:
        return ExpandedResolverRecord(
            record_id=record.record_id,
            source_context_id=record.source_context_id,
            conversation_id=record.conversation_id,
            split=record.split,
            skill=record.skill,
            case=record.case,
            prompt=record.prompt,
            expected_action=GENERATE_ACTION,
            expected_type=None,
            candidates=(),
            selected_candidate_index=None,
            response_template=None,
            expected_value=None,
            alternative_value=None,
            source_phase="phase34",
        )
    return ExpandedResolverRecord(
        record_id=record.record_id,
        source_context_id=record.source_context_id,
        conversation_id=record.conversation_id,
        split=record.split,
        skill=record.skill,
        case=record.case,
        prompt=record.prompt,
        expected_action=CLARIFY_ACTION,
        expected_type=record.expected_type,
        candidates=proposed_candidates(record, spans),
        selected_candidate_index=None,
        response_template=CLARIFICATION_RESPONSE,
        expected_value=record.expected_value,
        alternative_value=record.alternative_value,
        source_phase="phase34",
    )


def realize_proposed_decision(
    source_record: ExpandedResolverRecord,
    runtime_record: ExpandedResolverRecord,
    mode_index: int,
    option_index: int,
) -> ExpandedResolverDecision:
    """Realize a proposal-backed option against the source task frame."""

    if mode_index == MODE_TO_INDEX[GENERATE_MODE]:
        return ExpandedResolverDecision(GENERATE_ACTION, None, None, None)
    if option_index == NO_SUPPORT_OPTION_INDEX:
        return ExpandedResolverDecision(
            CLARIFY_ACTION, CLARIFICATION_RESPONSE, None, None
        )
    valid = (
        0 <= option_index < len(runtime_record.candidates)
        and source_record.expected_type is not None
        and runtime_record.candidates[option_index].value_type
        == source_record.expected_type
    )
    if not valid:
        return ExpandedResolverDecision(
            CLARIFY_ACTION,
            CLARIFICATION_RESPONSE,
            None,
            None,
            guarded=True,
        )
    candidate = runtime_record.candidates[option_index]
    frame = SKILL_RESPONSE_FRAMES[source_record.skill]
    return ExpandedResolverDecision(
        RESOLVE_ACTION,
        frame.replace("<|resolved_value|>", candidate.text),
        option_index,
        candidate.text,
    )


def proposal_metrics(rows: Iterable[dict]) -> dict:
    rows = tuple(rows)
    gold = sum(int(row["gold_span_count"]) for row in rows)
    predicted = sum(int(row["predicted_span_count"]) for row in rows)
    exact = sum(int(row["exact_span_matches"]) for row in rows)
    boundary = sum(int(row["boundary_matches"]) for row in rows)
    resolve = tuple(row for row in rows if row["expected_action"] == RESOLVE_ACTION)
    return {
        "examples": len(rows),
        "gold_spans": gold,
        "predicted_spans": predicted,
        "exact_span_precision": (
            exact / predicted if predicted else (1.0 if not gold else 0.0)
        ),
        "exact_span_recall": exact / gold if gold else 1.0,
        "exact_span_f1": (
            2 * exact / (gold + predicted) if gold + predicted else 1.0
        ),
        "boundary_type_accuracy": (
            exact / boundary if boundary else (1.0 if not gold else 0.0)
        ),
        "answer_candidate_recall": (
            sum(bool(row["answer_candidate_present"]) for row in resolve)
            / len(resolve)
            if resolve
            else 1.0
        ),
        "offset_validity_rate": (
            sum(bool(row["offsets_valid"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "proposal_overflow_rate": (
            sum(bool(row["proposal_overflow"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
    }


def proposal_metric_row(
    record: ExpandedResolverRecord,
    gold_spans: Iterable[EvidenceSpan],
    predicted_spans: Iterable[EvidenceSpan],
) -> dict:
    gold_spans = tuple(gold_spans)
    predicted_spans = tuple(predicted_spans)
    offsets_valid = True
    try:
        for span in predicted_spans:
            span.validate_prompt(record.prompt)
    except ValueError:
        offsets_valid = False
    gold_exact = {
        (span.byte_start, span.byte_end, span.value_type) for span in gold_spans
    }
    predicted_exact = {
        (span.byte_start, span.byte_end, span.value_type)
        for span in predicted_spans
    }
    gold_boundaries = {(span.byte_start, span.byte_end) for span in gold_spans}
    predicted_boundaries = {
        (span.byte_start, span.byte_end) for span in predicted_spans
    }
    candidates = proposed_candidates(record, predicted_spans)
    return {
        "record_id": record.record_id,
        "conversation_id": record.conversation_id,
        "split": record.split,
        "skill": record.skill,
        "case": record.case,
        "expected_action": record.expected_action,
        "expected_value": record.expected_value,
        "gold_span_count": len(gold_exact),
        "predicted_span_count": len(predicted_exact),
        "exact_span_matches": len(gold_exact & predicted_exact),
        "boundary_matches": len(gold_boundaries & predicted_boundaries),
        "answer_candidate_present": (
            record.expected_action != RESOLVE_ACTION
            or any(
                candidate.text == record.expected_value
                and candidate.value_type == record.expected_type
                for candidate in candidates
            )
        ),
        "offsets_valid": offsets_valid,
        "proposal_overflow": (
            valid_unique_proposed_span_count(record, predicted_spans)
            > MAX_CANDIDATES
        ),
        "gold_spans": [span.to_dict() for span in gold_spans],
        "predicted_spans": [span.to_dict() for span in predicted_spans],
        "proposed_candidates": [candidate.to_dict() for candidate in candidates],
    }
