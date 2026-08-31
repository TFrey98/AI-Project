"""Phase 31 typed whole-span selection and deterministic realization."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from story_model.character_data import (
    CHARACTER_CONTROL_TOKENS,
    ConversationTurn,
    serialize_character_prompt,
)
from story_model.character_training import CharacterTrainingRecord
from story_model.corpus import sha256_text
from story_model.data import ByteBPETokenizer
from story_model.neutral_diagnostics import neutral_skill


TYPED_SPAN_RESOLVER_VERSION = 1
TYPED_SPAN_SPLITS = (
    "train",
    "val",
    "lexical",
    "paraphrase",
    "transfer",
)

RESOLVE_ACTION = "resolve"
CLARIFY_ACTION = "clarify"
GENERATE_ACTION = "generate"
RESOLVER_ACTIONS = (
    RESOLVE_ACTION,
    CLARIFY_ACTION,
    GENERATE_ACTION,
)
ACTION_TO_INDEX = {
    action: index for index, action in enumerate(RESOLVER_ACTIONS)
}

RESOLVER_MARKER = "<|resolver|>"
CANDIDATE_MARKER = "<|candidate|>"
DECISION_MARKER = "<|decision|>"
RESOLVED_VALUE_MARKER = "<|resolved_value|>"
RESOLVER_CONTROL_TOKENS = (
    RESOLVER_MARKER,
    CANDIDATE_MARKER,
    DECISION_MARKER,
    RESOLVED_VALUE_MARKER,
)
TYPED_SPAN_CONTROL_TOKENS = (
    *CHARACTER_CONTROL_TOKENS,
    *RESOLVER_CONTROL_TOKENS,
)

EXPECTED_TYPES = {
    "supplied_fact": "color",
    "scene_route": "route",
}
RESPONSE_FRAMES = {
    "supplied_fact": (
        "The supplied evidence identifies "
        f"{RESOLVED_VALUE_MARKER}."
    ),
    "scene_route": f"Take the {RESOLVED_VALUE_MARKER}.",
}
CLARIFICATION_RESPONSE = (
    "I cannot resolve that from the supplied evidence."
)
GENERATE_REQUESTS = (
    "Acknowledge that you heard me; do not resolve a typed value.",
    "Respond normally without selecting a color or route.",
    "For now, simply confirm that you understand the request.",
    "Do not answer the earlier factual question; just acknowledge it.",
)
WRONG_TYPE_CANDIDATES = (
    ("puppies", "animal"),
    ("kittens", "animal"),
)


def _clean_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{label} cannot be empty")
    return cleaned


@dataclass(frozen=True)
class TypedSpanCandidate:
    """One exact candidate supplied by the span annotation layer."""

    text: str
    value_type: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "text", _clean_text(self.text, "candidate text")
        )
        object.__setattr__(
            self,
            "value_type",
            _clean_text(self.value_type, "candidate value_type"),
        )
        if "<|" in self.text:
            raise ValueError("candidate text cannot contain control markers")

    def to_dict(self) -> dict:
        return {"text": self.text, "value_type": self.value_type}

    @classmethod
    def from_dict(cls, data: dict) -> "TypedSpanCandidate":
        if not isinstance(data, dict):
            raise TypeError("candidate must be an object")
        return cls(text=data["text"], value_type=data["value_type"])


@dataclass(frozen=True)
class TypedSpanRecord:
    """One structured routing and whole-candidate selection example."""

    record_id: str
    source_context_id: str
    conversation_id: str
    split: str
    skill: str
    case: str
    prompt: str
    expected_action: str
    expected_type: Optional[str]
    candidates: tuple[TypedSpanCandidate, ...]
    selected_candidate_index: Optional[int]
    response_template: Optional[str]
    expected_value: Optional[str]
    alternative_value: Optional[str]

    def __post_init__(self) -> None:
        for attribute in (
            "record_id",
            "source_context_id",
            "conversation_id",
            "split",
            "skill",
            "case",
            "prompt",
            "expected_action",
        ):
            object.__setattr__(
                self,
                attribute,
                _clean_text(getattr(self, attribute), attribute),
            )

        if self.split not in TYPED_SPAN_SPLITS:
            raise ValueError(f"unknown typed-span split: {self.split}")
        if self.skill not in EXPECTED_TYPES:
            raise ValueError(f"unknown typed-span skill: {self.skill}")
        if self.expected_action not in RESOLVER_ACTIONS:
            raise ValueError(
                f"unknown resolver action: {self.expected_action}"
            )
        if not isinstance(self.candidates, tuple) or not all(
            isinstance(candidate, TypedSpanCandidate)
            for candidate in self.candidates
        ):
            raise TypeError("candidates must be typed-span candidates")
        if len(self.candidates) > 2:
            raise ValueError("Phase 31 supports at most two candidates")
        if len({candidate.text for candidate in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("candidate texts cannot repeat")

        if self.expected_action == RESOLVE_ACTION:
            if self.expected_type is None:
                raise ValueError("resolve records require expected_type")
            if self.selected_candidate_index is None:
                raise ValueError("resolve records require a selected candidate")
            if not 0 <= self.selected_candidate_index < len(self.candidates):
                raise ValueError("selected candidate index is out of range")
            selected = self.candidates[self.selected_candidate_index]
            if selected.value_type != self.expected_type:
                raise ValueError("selected candidate has the wrong type")
            if selected.text != self.expected_value:
                raise ValueError(
                    "selected candidate does not match expected_value"
                )
            if (
                self.response_template is None
                or self.response_template.count(RESOLVED_VALUE_MARKER) != 1
            ):
                raise ValueError(
                    "resolve response_template must contain one placeholder"
                )
        else:
            if self.selected_candidate_index is not None:
                raise ValueError(
                    "non-resolve records cannot select a candidate"
                )
            if self.response_template is not None and (
                RESOLVED_VALUE_MARKER in self.response_template
            ):
                raise ValueError(
                    "non-resolve templates cannot contain the placeholder"
                )

        if self.expected_action == GENERATE_ACTION:
            if self.expected_type is not None or self.candidates:
                raise ValueError(
                    "generate records cannot require a type or candidates"
                )

    def to_dict(self) -> dict:
        return {
            "resolver_version": TYPED_SPAN_RESOLVER_VERSION,
            "record_id": self.record_id,
            "source_context_id": self.source_context_id,
            "conversation_id": self.conversation_id,
            "split": self.split,
            "skill": self.skill,
            "case": self.case,
            "prompt": self.prompt,
            "expected_action": self.expected_action,
            "expected_type": self.expected_type,
            "candidates": [
                candidate.to_dict() for candidate in self.candidates
            ],
            "selected_candidate_index": self.selected_candidate_index,
            "response_template": self.response_template,
            "expected_value": self.expected_value,
            "alternative_value": self.alternative_value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TypedSpanRecord":
        if not isinstance(data, dict):
            raise TypeError("typed-span record must be an object")
        if data.get("resolver_version") != TYPED_SPAN_RESOLVER_VERSION:
            raise ValueError(
                "unsupported typed-span resolver version: "
                f"{data.get('resolver_version')!r}"
            )
        return cls(
            record_id=data["record_id"],
            source_context_id=data["source_context_id"],
            conversation_id=data["conversation_id"],
            split=data["split"],
            skill=data["skill"],
            case=data["case"],
            prompt=data["prompt"],
            expected_action=data["expected_action"],
            expected_type=data.get("expected_type"),
            candidates=tuple(
                TypedSpanCandidate.from_dict(candidate)
                for candidate in data.get("candidates", ())
            ),
            selected_candidate_index=data.get("selected_candidate_index"),
            response_template=data.get("response_template"),
            expected_value=data.get("expected_value"),
            alternative_value=data.get("alternative_value"),
        )


def typed_span_record_to_json(record: TypedSpanRecord) -> str:
    return json.dumps(
        record.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def save_typed_span_records(
    records: Iterable[TypedSpanRecord],
    path: Union[str, Path],
) -> None:
    records = tuple(records)
    if not records:
        raise ValueError("typed-span dataset cannot be empty")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("typed-span record_id values must be unique")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(typed_span_record_to_json(record) for record in records),
        encoding="utf-8",
    )


def load_typed_span_records(
    path: Union[str, Path],
) -> tuple[TypedSpanRecord, ...]:
    path = Path(path)
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL line")
        try:
            records.append(TypedSpanRecord.from_dict(json.loads(line)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    records = tuple(records)
    if not records:
        raise ValueError("typed-span dataset cannot be empty")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("typed-span record_id values must be unique")
    return records


def _context_without_decisive_evidence(record: CharacterTrainingRecord):
    """Remove only the counterfactual evidence while preserving the task."""

    context = record.context
    changed = False
    world_facts = tuple(
        fact
        for fact in context.world_facts
        if not fact.fact_id.endswith("_evidence")
    )
    if world_facts != context.world_facts:
        changed = True
    memories = tuple(
        memory
        for memory in context.memories
        if not memory.memory_id.endswith("_evidence")
    )
    if memories != context.memories:
        changed = True

    situation = context.scene.situation
    if " Evidence: " in situation:
        situation = situation.split(" Evidence: ", 1)[0]
        changed = True

    recent_turns = context.recent_turns
    if len(recent_turns) >= 3 and (
        recent_turns[-2].role == "assistant"
        and recent_turns[-3].text
        == "Review the supplied evidence before I ask."
    ):
        recent_turns = (recent_turns[-1],)
        changed = True

    if not changed:
        raise ValueError(
            f"could not locate decisive evidence in {context.context_id}"
        )

    return replace(
        context,
        scene=replace(context.scene, situation=situation),
        world_facts=world_facts,
        memories=memories,
        recent_turns=recent_turns,
        target_response=None,
    )


def _generate_context(record: CharacterTrainingRecord, pair_index: int):
    context = record.context
    latest = context.recent_turns[-1]
    generated_turn = ConversationTurn(
        role="user",
        speaker_id=latest.speaker_id,
        text=GENERATE_REQUESTS[pair_index % len(GENERATE_REQUESTS)],
    )
    return replace(
        context,
        recent_turns=(*context.recent_turns[:-1], generated_turn),
        target_response=None,
    )


def _answer_key_entry(answer_keys: dict, context_id: str) -> dict:
    entries = answer_keys.get("entries")
    if not isinstance(entries, dict) or context_id not in entries:
        raise ValueError(f"missing answer key for {context_id}")
    entry = entries[context_id]
    if not isinstance(entry, dict):
        raise TypeError(f"answer key for {context_id} must be an object")
    return entry


def _resolve_record(
    record: CharacterTrainingRecord,
    answer_keys: dict,
    split: str,
) -> TypedSpanRecord:
    skill = neutral_skill(record)
    entry = _answer_key_entry(answer_keys, record.context.context_id)
    expected = _clean_text(entry["expected_value"], "expected value")
    alternative = _clean_text(
        entry["alternative_value"], "alternative value"
    )
    expected_type = EXPECTED_TYPES[skill]
    side = int(record.context.context_id.rsplit("_", 1)[-1]) % 2

    # Across a pair this gives an identical inventory but flips the label.
    ordered_values = (
        (expected, alternative) if side == 0 else (alternative, expected)
    )
    selected_index = 0 if side == 0 else 1
    candidates = tuple(
        TypedSpanCandidate(text=value, value_type=expected_type)
        for value in ordered_values
    )
    return TypedSpanRecord(
        record_id=f"{record.context.context_id}:resolve",
        source_context_id=record.context.context_id,
        conversation_id=record.conversation_id,
        split=split,
        skill=skill,
        case="supported",
        prompt=serialize_character_prompt(
            replace(record.context, target_response=None)
        ),
        expected_action=RESOLVE_ACTION,
        expected_type=expected_type,
        candidates=candidates,
        selected_candidate_index=selected_index,
        response_template=RESPONSE_FRAMES[skill],
        expected_value=expected,
        alternative_value=alternative,
    )


def build_typed_span_records(
    source_records: Iterable[CharacterTrainingRecord],
    answer_keys: dict,
    split: str,
) -> tuple[TypedSpanRecord, ...]:
    """Build resolve rows plus balanced clarify/generate controls."""

    source_records = tuple(source_records)
    if split not in TYPED_SPAN_SPLITS:
        raise ValueError(f"unknown typed-span split: {split}")
    if not source_records or len(source_records) % 2:
        raise ValueError("source records must contain complete pairs")

    output = []
    for offset in range(0, len(source_records), 2):
        first, second = source_records[offset : offset + 2]
        if first.conversation_id != second.conversation_id:
            raise ValueError("source counterfactual pairs must remain adjacent")
        first_resolve = _resolve_record(first, answer_keys, split)
        second_resolve = _resolve_record(second, answer_keys, split)
        if first_resolve.candidates != second_resolve.candidates:
            raise ValueError(
                "counterfactual pair candidate inventories must match"
            )
        if (
            first_resolve.selected_candidate_index
            == second_resolve.selected_candidate_index
        ):
            raise ValueError(
                "counterfactual pair selected candidate must flip"
            )
        output.extend((first_resolve, second_resolve))

        pair_index = offset // 2
        if pair_index % 2 == 0:
            clarify_prompt = serialize_character_prompt(
                _context_without_decisive_evidence(first)
            )
            clarify_candidates = first_resolve.candidates
            clarify_case = "missing_evidence"
        else:
            wrong_text, wrong_type = WRONG_TYPE_CANDIDATES[
                pair_index % len(WRONG_TYPE_CANDIDATES)
            ]
            clarify_prompt = first_resolve.prompt
            clarify_candidates = (
                TypedSpanCandidate(wrong_text, wrong_type),
            )
            clarify_case = "wrong_type"

        output.append(
            TypedSpanRecord(
                record_id=f"{first.context.context_id}:clarify",
                source_context_id=first.context.context_id,
                conversation_id=f"{first.conversation_id}:clarify",
                split=split,
                skill=first_resolve.skill,
                case=clarify_case,
                prompt=clarify_prompt,
                expected_action=CLARIFY_ACTION,
                expected_type=first_resolve.expected_type,
                candidates=clarify_candidates,
                selected_candidate_index=None,
                response_template=CLARIFICATION_RESPONSE,
                expected_value=first_resolve.expected_value,
                alternative_value=first_resolve.alternative_value,
            )
        )
        output.append(
            TypedSpanRecord(
                record_id=f"{first.context.context_id}:generate",
                source_context_id=first.context.context_id,
                conversation_id=f"{first.conversation_id}:generate",
                split=split,
                skill=first_resolve.skill,
                case="ordinary_generation",
                prompt=serialize_character_prompt(
                    _generate_context(first, pair_index)
                ),
                expected_action=GENERATE_ACTION,
                expected_type=None,
                candidates=(),
                selected_candidate_index=None,
                response_template=None,
                expected_value=None,
                alternative_value=None,
            )
        )

    return tuple(output)


def validate_typed_span_records(
    records: Iterable[TypedSpanRecord],
) -> dict:
    records = tuple(records)
    if not records:
        raise ValueError("typed-span records cannot be empty")
    ids = [record.record_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("typed-span record_id values must be unique")

    actions = Counter(record.expected_action for record in records)
    cases = Counter(record.case for record in records)
    skills = Counter(record.skill for record in records)
    resolve_records = tuple(
        record
        for record in records
        if record.expected_action == RESOLVE_ACTION
    )
    selected = Counter(
        record.selected_candidate_index for record in resolve_records
    )
    if selected.get(0, 0) != selected.get(1, 0):
        raise ValueError("resolve candidate targets must be position-balanced")

    conversations = defaultdict(list)
    for record in resolve_records:
        conversations[record.conversation_id].append(record)
    if not conversations or not all(
        len(pair) == 2 for pair in conversations.values()
    ):
        raise ValueError("every resolve conversation must contain two rows")
    for pair in conversations.values():
        if pair[0].candidates != pair[1].candidates:
            raise ValueError("resolve pair candidate inventories changed")
        if (
            pair[0].selected_candidate_index
            == pair[1].selected_candidate_index
        ):
            raise ValueError("resolve pair target did not flip")

    return {
        "examples": len(records),
        "resolve_pairs": len(conversations),
        "actions": dict(sorted(actions.items())),
        "cases": dict(sorted(cases.items())),
        "skills": dict(sorted(skills.items())),
        "selected_candidate_positions": {
            str(key): value for key, value in sorted(selected.items())
        },
    }


def build_typed_span_dataset(
    source_splits: dict[str, Iterable[CharacterTrainingRecord]],
    answer_keys: dict,
    output_dir: Union[str, Path],
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "resolver_version": TYPED_SPAN_RESOLVER_VERSION,
        "split_strategy": (
            "Phase 30 resolve rows plus one clarify and one generate "
            "control per counterfactual pair"
        ),
        "splits": {},
    }
    for split in TYPED_SPAN_SPLITS:
        if split not in source_splits:
            raise ValueError(f"source splits are missing {split}")
        records = build_typed_span_records(
            source_splits[split], answer_keys, split
        )
        report = validate_typed_span_records(records)
        path = output_dir / f"{split}.jsonl"
        save_typed_span_records(records, path)
        text = path.read_text(encoding="utf-8")
        manifest["splits"][split] = {
            **report,
            "path": str(path),
            "sha256": sha256_text(text),
        }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return manifest


@dataclass(frozen=True)
class EncodedTypedSpanExample:
    record_id: str
    input_ids: tuple[int, ...]
    sequence_tokens: int
    candidate_token_masks: tuple[tuple[bool, ...], ...]
    candidate_view_input_ids: tuple[tuple[int, ...], ...]
    candidate_view_sequence_tokens: tuple[int, ...]
    action_target: int
    candidate_target: int


def _require_control_tokens(tokenizer: ByteBPETokenizer) -> None:
    required = set(TYPED_SPAN_CONTROL_TOKENS)
    if not required.issubset(tokenizer.special_token_ids):
        missing = sorted(required - set(tokenizer.special_token_ids))
        raise ValueError(
            "typed-span tokenizer is missing control tokens: "
            + ", ".join(missing)
        )


def _padded(tokens: list[int], block_size: int, label: str) -> tuple[int, ...]:
    if len(tokens) > block_size:
        raise ValueError(
            f"{label} needs {len(tokens)} tokens but block_size is "
            f"{block_size}"
        )
    return tuple(tokens + [0] * (block_size - len(tokens)))


def encode_typed_span_record(
    record: TypedSpanRecord,
    tokenizer: ByteBPETokenizer,
    block_size: int,
    max_candidates: int = 2,
) -> EncodedTypedSpanExample:
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise TypeError("typed-span resolution requires byte-BPE")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if len(record.candidates) > max_candidates:
        raise ValueError("record exceeds max_candidates")
    _require_control_tokens(tokenizer)

    tokens = tokenizer.encode(record.prompt)
    tokens.extend(
        tokenizer.encode(
            f"{RESOLVER_MARKER}\nexpected_type: "
            f"{record.expected_type or 'none'}\n"
        )
    )
    spans = []
    for index, candidate in enumerate(record.candidates):
        tokens.extend(
            tokenizer.encode(
                f"{CANDIDATE_MARKER}\nindex: {index}; "
                f"type: {candidate.value_type}; value: "
            )
        )
        start = len(tokens)
        candidate_tokens = tokenizer.encode(candidate.text)
        if not candidate_tokens:
            raise ValueError("candidate encoded to no tokens")
        tokens.extend(candidate_tokens)
        spans.append((start, len(tokens)))
        tokens.extend(tokenizer.encode("\n"))
    tokens.extend(tokenizer.encode(f"{DECISION_MARKER}\n"))

    sequence_tokens = len(tokens)
    input_ids = _padded(tokens, block_size, record.record_id)
    masks = []
    for start, end in spans:
        masks.append(
            tuple(start <= position < end for position in range(block_size))
        )
    while len(masks) < max_candidates:
        masks.append(tuple(False for _ in range(block_size)))

    # Each candidate is scored in a shared view.  Its exact text is replaced
    # by one atomic marker, so the classifier learns evidence relations rather
    # than memorizing value spelling or candidate position.
    candidate_views = []
    candidate_view_lengths = []
    for candidate in record.candidates:
        marked_prompt = record.prompt.replace(
            candidate.text, CANDIDATE_MARKER
        )
        view_tokens = tokenizer.encode(marked_prompt)
        view_tokens.extend(
            tokenizer.encode(
                f"{RESOLVER_MARKER}\n"
                f"expected_type: {record.expected_type or 'none'}\n"
                f"candidate_type: {candidate.value_type}\n"
                f"{DECISION_MARKER}\n"
            )
        )
        candidate_view_lengths.append(len(view_tokens))
        candidate_views.append(
            _padded(
                view_tokens,
                block_size,
                f"candidate view for {record.record_id}",
            )
        )
    while len(candidate_views) < max_candidates:
        candidate_views.append(tuple(0 for _ in range(block_size)))
        candidate_view_lengths.append(1)

    return EncodedTypedSpanExample(
        record_id=record.record_id,
        input_ids=input_ids,
        sequence_tokens=sequence_tokens,
        candidate_token_masks=tuple(masks),
        candidate_view_input_ids=tuple(candidate_views),
        candidate_view_sequence_tokens=tuple(candidate_view_lengths),
        action_target=ACTION_TO_INDEX[record.expected_action],
        candidate_target=(
            -100
            if record.selected_candidate_index is None
            else record.selected_candidate_index
        ),
    )


def encode_typed_span_records(
    records: Iterable[TypedSpanRecord],
    tokenizer: ByteBPETokenizer,
    block_size: int,
) -> tuple[EncodedTypedSpanExample, ...]:
    return tuple(
        encode_typed_span_record(record, tokenizer, block_size)
        for record in records
    )


def typed_span_batch(
    examples: tuple[EncodedTypedSpanExample, ...],
    indices: Iterable[int],
    device: Union[str, torch.device],
) -> tuple[torch.Tensor, ...]:
    selected = tuple(examples[index] for index in indices)
    if not selected:
        raise ValueError("typed-span batch cannot be empty")
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
            [example.candidate_token_masks for example in selected],
            dtype=torch.bool,
            device=device,
        ),
        torch.tensor(
            [example.candidate_view_input_ids for example in selected],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [
                example.candidate_view_sequence_tokens
                for example in selected
            ],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [example.action_target for example in selected],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [example.candidate_target for example in selected],
            dtype=torch.long,
            device=device,
        ),
    )


@dataclass
class TypedSpanResolverOutput:
    action_logits: torch.Tensor
    candidate_logits: torch.Tensor
    loss: Optional[torch.Tensor]
    action_loss: Optional[torch.Tensor]
    candidate_loss: Optional[torch.Tensor]


class TypedSpanResolver(nn.Module):
    """Classifier heads over a warm-started decoder Transformer."""

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
                f"Sequence length {sequence} exceeds block size "
                f"{self.block_size}"
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
        candidate_token_masks: torch.Tensor,
        candidate_view_tokens: torch.Tensor,
        candidate_view_sequence_lengths: torch.Tensor,
        action_targets: Optional[torch.Tensor] = None,
        candidate_targets: Optional[torch.Tensor] = None,
    ) -> TypedSpanResolverOutput:
        batch, candidate_count, sequence = candidate_view_tokens.shape
        combined_tokens = torch.cat(
            (tokens, candidate_view_tokens.reshape(-1, sequence)), dim=0
        )
        combined_hidden = self.hidden_states(combined_tokens)
        hidden = combined_hidden[:batch]
        candidate_hidden = combined_hidden[batch:].reshape(
            batch, candidate_count, sequence, -1
        )

        query_positions = sequence_lengths - 1
        queries = hidden[
            torch.arange(batch, device=tokens.device), query_positions
        ]
        action_logits = self.action_head(queries)

        candidate_positions = candidate_view_sequence_lengths - 1
        batch_indices = torch.arange(batch, device=tokens.device).unsqueeze(1)
        candidate_indices = torch.arange(
            candidate_count, device=tokens.device
        ).unsqueeze(0)
        candidate_states = candidate_hidden[
            batch_indices,
            candidate_indices,
            candidate_positions,
        ]
        candidate_logits = self.candidate_score(candidate_states).squeeze(-1)
        valid_candidates = candidate_token_masks.any(dim=-1)
        candidate_logits = candidate_logits.masked_fill(
            ~valid_candidates, torch.finfo(candidate_logits.dtype).min
        )

        action_loss = None
        candidate_loss = None
        loss = None
        if action_targets is not None:
            action_loss = F.cross_entropy(action_logits, action_targets)
            loss = action_loss
        if candidate_targets is not None:
            resolve_rows = candidate_targets != -100
            if resolve_rows.any():
                candidate_loss = F.cross_entropy(
                    candidate_logits[resolve_rows],
                    candidate_targets[resolve_rows],
                )
                loss = candidate_loss if loss is None else loss + candidate_loss

        return TypedSpanResolverOutput(
            action_logits=action_logits,
            candidate_logits=candidate_logits,
            loss=loss,
            action_loss=action_loss,
            candidate_loss=candidate_loss,
        )


@dataclass(frozen=True)
class ResolverDecision:
    action: str
    text: Optional[str]
    selected_candidate_index: Optional[int]
    selected_value: Optional[str]
    guarded: bool = False


def realize_resolver_decision(
    record: TypedSpanRecord,
    action: str,
    candidate_index: Optional[int],
) -> ResolverDecision:
    """Apply an action while failing closed on invalid candidate choices."""

    if action not in RESOLVER_ACTIONS:
        raise ValueError(f"unknown resolver action: {action}")
    if action == GENERATE_ACTION:
        return ResolverDecision(action, None, None, None)
    if action == CLARIFY_ACTION:
        return ResolverDecision(
            action, CLARIFICATION_RESPONSE, None, None
        )

    valid = (
        candidate_index is not None
        and 0 <= candidate_index < len(record.candidates)
        and record.expected_type is not None
        and record.candidates[candidate_index].value_type
        == record.expected_type
        and record.response_template is not None
        and record.response_template.count(RESOLVED_VALUE_MARKER) == 1
    )
    if not valid:
        return ResolverDecision(
            CLARIFY_ACTION,
            CLARIFICATION_RESPONSE,
            None,
            None,
            guarded=True,
        )

    assert candidate_index is not None
    assert record.response_template is not None
    candidate = record.candidates[candidate_index]
    text = record.response_template.replace(
        RESOLVED_VALUE_MARKER, candidate.text
    )
    return ResolverDecision(
        RESOLVE_ACTION,
        text,
        candidate_index,
        candidate.text,
    )


def summarize_resolver_predictions(rows: Iterable[dict]) -> dict:
    rows = tuple(rows)
    if not rows:
        raise ValueError("resolver prediction rows cannot be empty")

    def rate(predicate) -> float:
        return sum(bool(predicate(row)) for row in rows) / len(rows)

    resolve_rows = tuple(
        row for row in rows if row["expected_action"] == RESOLVE_ACTION
    )
    clarify_rows = tuple(
        row for row in rows if row["expected_action"] == CLARIFY_ACTION
    )
    generate_rows = tuple(
        row for row in rows if row["expected_action"] == GENERATE_ACTION
    )

    conversations = defaultdict(list)
    for row in resolve_rows:
        conversations[row["conversation_id"]].append(row)

    def subset_rate(subset, key: str) -> float:
        if not subset:
            return 0.0
        return sum(bool(row[key]) for row in subset) / len(subset)

    return {
        "examples": len(rows),
        "action_accuracy": rate(lambda row: row["action_correct"]),
        "resolve_examples": len(resolve_rows),
        "resolve_action_accuracy": subset_rate(
            resolve_rows, "action_correct"
        ),
        "end_to_end_resolve_accuracy": subset_rate(
            resolve_rows, "end_to_end_correct"
        ),
        "counterfactual_pair_resolve_accuracy": (
            sum(
                len(pair) == 2
                and all(row["end_to_end_correct"] for row in pair)
                for pair in conversations.values()
            )
            / len(conversations)
            if conversations
            else 0.0
        ),
        "exact_realization_rate": subset_rate(
            resolve_rows, "exact_realization"
        ),
        "value_missing_rate": subset_rate(resolve_rows, "value_missing"),
        "wrong_alternative_rate": subset_rate(
            resolve_rows, "wrong_alternative"
        ),
        "clarify_examples": len(clarify_rows),
        "clarify_accuracy": subset_rate(clarify_rows, "action_correct"),
        "generate_examples": len(generate_rows),
        "generate_accuracy": subset_rate(generate_rows, "action_correct"),
        "guarded_rate": rate(lambda row: row["guarded"]),
        "cases": dict(sorted(Counter(row["case"] for row in rows).items())),
        "skills": dict(
            sorted(Counter(row["skill"] for row in rows).items())
        ),
    }
