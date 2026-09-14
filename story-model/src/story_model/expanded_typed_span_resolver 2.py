"""Phase 32 multi-candidate expansion of typed whole-span resolution."""

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
from story_model.corpus import sha256_text
from story_model.data import ByteBPETokenizer
from story_model.neutral_instruction import (
    TRAIN_LEXICON,
    VALIDATION_LEXICON,
    NeutralLexicon,
    neutral_instruction_skill_records,
)


EXPANDED_RESOLVER_VERSION = 1
EXPANDED_SPLITS = ("train", "val", "lexical", "paraphrase", "transfer")
DIRECT_SPAN_SKILLS = (
    "multi_turn_memory",
    "contradiction_correction",
    "reference_tracking",
    "promise_recall",
)
SKILL_VALUE_TYPES = {
    "scene_route": "route",
    "supplied_fact": "color",
    "multi_turn_memory": "container",
    "contradiction_correction": "person",
    "reference_tracking": "container",
    "promise_recall": "action",
}
SKILL_RESPONSE_FRAMES = {
    "scene_route": "Take the <|resolved_value|>.",
    "supplied_fact": "The supplied evidence identifies <|resolved_value|>.",
    "multi_turn_memory": "The recorded location is <|resolved_value|>.",
    "contradiction_correction": "The current holder is <|resolved_value|>.",
    "reference_tracking": "The object's current location is <|resolved_value|>.",
    "promise_recall": "The recorded commitment is to <|resolved_value|>.",
}

RESOLVE_ACTION = "resolve"
CLARIFY_ACTION = "clarify"
GENERATE_ACTION = "generate"
RESOLVER_ACTIONS = (RESOLVE_ACTION, CLARIFY_ACTION, GENERATE_ACTION)
ACTION_TO_INDEX = {action: index for index, action in enumerate(RESOLVER_ACTIONS)}

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
EXPANDED_CONTROL_TOKENS = (*CHARACTER_CONTROL_TOKENS, *RESOLVER_CONTROL_TOKENS)
MAX_CANDIDATES = 4
CLARIFICATION_RESPONSE = "I cannot resolve that from the supplied evidence."
GENERATE_REQUESTS = (
    "Acknowledge that you heard me; do not resolve a typed value.",
    "Respond normally without selecting a stored value.",
    "For now, simply confirm that you understand the request.",
    "Do not answer the earlier question; just acknowledge it.",
)
WRONG_TYPE_CANDIDATES = (("puppies", "animal"), ("kittens", "animal"))


def _clean_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{label} cannot be empty")
    return cleaned


@dataclass(frozen=True)
class ExpandedCandidate:
    text: str
    value_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _clean_text(self.text, "candidate text"))
        object.__setattr__(
            self, "value_type", _clean_text(self.value_type, "candidate value_type")
        )
        if "<|" in self.text:
            raise ValueError("candidate text cannot contain control markers")

    def to_dict(self) -> dict:
        return {"text": self.text, "value_type": self.value_type}

    @classmethod
    def from_dict(cls, data: dict) -> "ExpandedCandidate":
        return cls(text=data["text"], value_type=data["value_type"])


@dataclass(frozen=True)
class ExpandedResolverRecord:
    record_id: str
    source_context_id: str
    conversation_id: str
    split: str
    skill: str
    case: str
    prompt: str
    expected_action: str
    expected_type: Optional[str]
    candidates: tuple[ExpandedCandidate, ...]
    selected_candidate_index: Optional[int]
    response_template: Optional[str]
    expected_value: Optional[str]
    alternative_value: Optional[str]
    source_phase: str = "phase32"

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
            "source_phase",
        ):
            object.__setattr__(
                self, attribute, _clean_text(getattr(self, attribute), attribute)
            )
        if self.split not in EXPANDED_SPLITS:
            raise ValueError(f"unknown expanded resolver split: {self.split}")
        if self.skill not in SKILL_VALUE_TYPES:
            raise ValueError(f"unknown expanded resolver skill: {self.skill}")
        if self.expected_action not in RESOLVER_ACTIONS:
            raise ValueError(f"unknown resolver action: {self.expected_action}")
        if not isinstance(self.candidates, tuple) or not all(
            isinstance(candidate, ExpandedCandidate) for candidate in self.candidates
        ):
            raise TypeError("candidates must be ExpandedCandidate values")
        if len(self.candidates) > MAX_CANDIDATES:
            raise ValueError(f"Phase 32 supports at most {MAX_CANDIDATES} candidates")
        if len({candidate.text for candidate in self.candidates}) != len(
            self.candidates
        ):
            raise ValueError("candidate texts cannot repeat")

        if self.expected_action == RESOLVE_ACTION:
            if self.expected_type is None or not self.candidates:
                raise ValueError("resolve records require a type and candidates")
            if self.selected_candidate_index is None or not (
                0 <= self.selected_candidate_index < len(self.candidates)
            ):
                raise ValueError("resolve record selected index is invalid")
            selected = self.candidates[self.selected_candidate_index]
            if selected.value_type != self.expected_type:
                raise ValueError("selected candidate has the wrong type")
            if selected.text != self.expected_value:
                raise ValueError("selected candidate does not match expected_value")
            if (
                self.response_template is None
                or self.response_template.count(RESOLVED_VALUE_MARKER) != 1
            ):
                raise ValueError("resolve template requires one resolved placeholder")
            if self.expected_value not in self.prompt:
                raise ValueError("resolve prompt does not contain expected_value")
            if self.source_phase == "phase32":
                present = tuple(
                    candidate.text
                    for candidate in self.candidates
                    if candidate.text in self.prompt
                )
                if present != (self.expected_value,):
                    raise ValueError(
                        "Phase 32 resolve prompt must contain exactly the "
                        "expected candidate"
                    )
        else:
            if self.selected_candidate_index is not None:
                raise ValueError("non-resolve records cannot select a candidate")
            if self.response_template and RESOLVED_VALUE_MARKER in self.response_template:
                raise ValueError("non-resolve template cannot contain the placeholder")
        if self.expected_action == GENERATE_ACTION and (
            self.expected_type is not None or self.candidates
        ):
            raise ValueError("generate records cannot require a type or candidates")

    def to_dict(self) -> dict:
        return {
            "expanded_resolver_version": EXPANDED_RESOLVER_VERSION,
            "record_id": self.record_id,
            "source_context_id": self.source_context_id,
            "conversation_id": self.conversation_id,
            "split": self.split,
            "skill": self.skill,
            "case": self.case,
            "prompt": self.prompt,
            "expected_action": self.expected_action,
            "expected_type": self.expected_type,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_candidate_index": self.selected_candidate_index,
            "response_template": self.response_template,
            "expected_value": self.expected_value,
            "alternative_value": self.alternative_value,
            "source_phase": self.source_phase,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExpandedResolverRecord":
        if not isinstance(data, dict):
            raise TypeError("expanded resolver record must be an object")
        phase31 = data.get("resolver_version") == 1
        phase32 = data.get("expanded_resolver_version") == EXPANDED_RESOLVER_VERSION
        if not (phase31 or phase32):
            raise ValueError("unsupported typed-span resolver record version")
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
                ExpandedCandidate.from_dict(candidate)
                for candidate in data.get("candidates", ())
            ),
            selected_candidate_index=data.get("selected_candidate_index"),
            response_template=data.get("response_template"),
            expected_value=data.get("expected_value"),
            alternative_value=data.get("alternative_value"),
            source_phase=("phase31" if phase31 else data.get("source_phase", "phase32")),
        )


def save_expanded_records(
    records: Iterable[ExpandedResolverRecord], path: Union[str, Path]
) -> None:
    records = tuple(records)
    if not records:
        raise ValueError("expanded resolver dataset cannot be empty")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("record_id values must be unique")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def load_expanded_records(
    path: Union[str, Path],
) -> tuple[ExpandedResolverRecord, ...]:
    path = Path(path)
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL line")
        try:
            records.append(ExpandedResolverRecord.from_dict(json.loads(line)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    records = tuple(records)
    if not records:
        raise ValueError("expanded resolver dataset cannot be empty")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("record_id values must be unique")
    return records


def _value_pool(skill: str, lexicon: NeutralLexicon) -> tuple[str, ...]:
    if skill in {"multi_turn_memory", "reference_tracking"}:
        return lexicon.containers
    if skill == "contradiction_correction":
        return lexicon.names
    if skill == "promise_recall":
        return lexicon.actions
    raise ValueError(f"no direct-span value pool for {skill}")


def _extract_expected_value(record, skill: str, lexicon: NeutralLexicon) -> str:
    target = record.context.target_response
    if not isinstance(target, str):
        raise ValueError("neutral source record has no target response")
    matches = [value for value in _value_pool(skill, lexicon) if value in target]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {skill} value in target, found {matches!r}: {target!r}"
        )
    return matches[0]


def _candidate_values(
    expected: str,
    prompt: str,
    pool: tuple[str, ...],
    count: int,
) -> tuple[str, ...]:
    values = [expected]
    for value in pool:
        if value != expected and value not in prompt and value not in values:
            values.append(value)
        if len(values) == count:
            break
    if len(values) != count:
        raise ValueError("not enough absent same-type candidates")
    return tuple(values)


def _rotated(values: tuple[str, ...], offset: int) -> tuple[str, ...]:
    shift = offset % len(values)
    return values[shift:] + values[:shift]


def _generate_prompt(record, pair_index: int) -> str:
    context = record.context
    latest = context.recent_turns[-1]
    generated_turn = ConversationTurn(
        role="user",
        speaker_id=latest.speaker_id,
        text=GENERATE_REQUESTS[pair_index % len(GENERATE_REQUESTS)],
    )
    return serialize_character_prompt(
        replace(
            context,
            recent_turns=(*context.recent_turns[:-1], generated_turn),
            target_response=None,
        )
    )


def build_expanded_records(
    source_records: Iterable,
    split: str,
    skill: str,
    lexicon: NeutralLexicon,
) -> tuple[ExpandedResolverRecord, ...]:
    """Make matched counterfactual pairs with 2--4 same-type candidates."""

    if split not in EXPANDED_SPLITS:
        raise ValueError(f"unknown expanded resolver split: {split}")
    if skill not in DIRECT_SPAN_SKILLS:
        raise ValueError(f"Phase 32 does not include skill {skill}")
    source_records = tuple(source_records)
    if not source_records:
        raise ValueError("source records cannot be empty")

    value_type = SKILL_VALUE_TYPES[skill]
    pool = _value_pool(skill, lexicon)
    output = []
    for pair_index, source in enumerate(source_records):
        if source.context.target_response is None:
            raise ValueError("source record is missing target response")
        prompt = serialize_character_prompt(
            replace(source.context, target_response=None)
        )
        expected = _extract_expected_value(source, skill, lexicon)
        if expected not in prompt:
            raise ValueError(
                f"source evidence does not contain expected value {expected!r}"
            )
        candidate_count = 2 + pair_index % (MAX_CANDIDATES - 1)
        unrotated = _candidate_values(expected, prompt, pool, candidate_count)
        alternative = unrotated[1]
        # Width cycles 2/3/4. Advancing rotation once per complete cycle
        # covers every target position at every width.
        ordered_values = _rotated(unrotated, pair_index // 3)
        candidates = tuple(
            ExpandedCandidate(value, value_type) for value in ordered_values
        )
        original_index = ordered_values.index(expected)
        alternative_index = ordered_values.index(alternative)
        conversation_id = f"expanded_{split}_{skill}_{pair_index:06d}"
        base = {
            "source_context_id": source.context.context_id,
            "conversation_id": conversation_id,
            "split": split,
            "skill": skill,
            "case": "supported",
            "expected_action": RESOLVE_ACTION,
            "expected_type": value_type,
            "candidates": candidates,
            "response_template": SKILL_RESPONSE_FRAMES[skill],
            "source_phase": "phase32",
        }
        output.extend(
            (
                ExpandedResolverRecord(
                    record_id=f"{conversation_id}:0",
                    prompt=prompt,
                    selected_candidate_index=original_index,
                    expected_value=expected,
                    alternative_value=alternative,
                    **base,
                ),
                ExpandedResolverRecord(
                    record_id=f"{conversation_id}:1",
                    prompt=prompt.replace(expected, alternative),
                    selected_candidate_index=alternative_index,
                    expected_value=alternative,
                    alternative_value=expected,
                    **base,
                ),
            )
        )

        if pair_index % 2 == 0:
            clarify_prompt = prompt.replace(expected, "an unavailable value")
            clarify_candidates = candidates
            clarify_case = "missing_evidence"
        else:
            wrong_text, wrong_type = WRONG_TYPE_CANDIDATES[
                pair_index % len(WRONG_TYPE_CANDIDATES)
            ]
            clarify_prompt = prompt
            clarify_candidates = (ExpandedCandidate(wrong_text, wrong_type),)
            clarify_case = "wrong_type"
        output.append(
            ExpandedResolverRecord(
                record_id=f"{conversation_id}:clarify",
                source_context_id=source.context.context_id,
                conversation_id=f"{conversation_id}:clarify",
                split=split,
                skill=skill,
                case=clarify_case,
                prompt=clarify_prompt,
                expected_action=CLARIFY_ACTION,
                expected_type=value_type,
                candidates=clarify_candidates,
                selected_candidate_index=None,
                response_template=CLARIFICATION_RESPONSE,
                expected_value=expected,
                alternative_value=alternative,
            )
        )
        output.append(
            ExpandedResolverRecord(
                record_id=f"{conversation_id}:generate",
                source_context_id=source.context.context_id,
                conversation_id=f"{conversation_id}:generate",
                split=split,
                skill=skill,
                case="ordinary_generation",
                prompt=_generate_prompt(source, pair_index),
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


def validate_expanded_records(records: Iterable[ExpandedResolverRecord]) -> dict:
    records = tuple(records)
    if not records:
        raise ValueError("expanded resolver records cannot be empty")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("record_id values must be unique")
    actions = Counter(record.expected_action for record in records)
    skills = Counter(record.skill for record in records)
    cases = Counter(record.case for record in records)
    candidate_counts = Counter(
        len(record.candidates)
        for record in records
        if record.expected_action == RESOLVE_ACTION
    )
    conversations = defaultdict(list)
    selected_positions = Counter()
    for record in records:
        if record.expected_action != RESOLVE_ACTION:
            continue
        conversations[record.conversation_id].append(record)
        selected_positions[
            (len(record.candidates), record.selected_candidate_index)
        ] += 1
    for pair in conversations.values():
        if len(pair) != 2:
            raise ValueError("every resolve conversation must contain two rows")
        if pair[0].candidates != pair[1].candidates:
            raise ValueError("counterfactual candidate inventory changed")
        if pair[0].selected_candidate_index == pair[1].selected_candidate_index:
            raise ValueError("counterfactual selected candidate did not flip")
    if set(candidate_counts) != {2, 3, 4}:
        raise ValueError("resolve curriculum must exercise 2, 3, and 4 candidates")
    for count in candidate_counts:
        represented = {
            position
            for candidate_count, position in selected_positions
            if candidate_count == count
        }
        if represented != set(range(count)):
            raise ValueError(f"candidate count {count} does not cover every position")
    return {
        "examples": len(records),
        "resolve_pairs": len(conversations),
        "actions": dict(sorted(actions.items())),
        "skills": dict(sorted(skills.items())),
        "cases": dict(sorted(cases.items())),
        "candidate_counts": {str(key): value for key, value in sorted(candidate_counts.items())},
        "selected_positions": {
            f"{count}:{position}": value
            for (count, position), value in sorted(selected_positions.items())
        },
    }


def expanded_source_splits(
    train_pairs_per_skill: int = 400,
    validation_pairs_per_skill: int = 100,
    lexical_pairs_per_skill: int = 100,
    paraphrase_pairs_per_skill: int = 100,
    transfer_pairs_per_skill: int = 100,
    seed: int = 1337,
) -> dict[str, dict[str, tuple]]:
    """Build factorized source axes for the direct-span family."""

    splits = {split: {} for split in EXPANDED_SPLITS}
    for skill_index, skill in enumerate(DIRECT_SPAN_SKILLS):
        skill_seed = seed + skill_index * 10_000
        familiar = neutral_instruction_skill_records(
            skill,
            "train",
            TRAIN_LEXICON,
            train_pairs_per_skill + validation_pairs_per_skill,
            skill_seed,
        )
        splits["train"][skill] = familiar[:train_pairs_per_skill]
        splits["val"][skill] = familiar[train_pairs_per_skill:]
        splits["lexical"][skill] = neutral_instruction_skill_records(
            skill,
            "train",
            VALIDATION_LEXICON,
            lexical_pairs_per_skill,
            skill_seed + 1,
        )
        splits["paraphrase"][skill] = neutral_instruction_skill_records(
            skill,
            "val",
            TRAIN_LEXICON,
            paraphrase_pairs_per_skill,
            skill_seed + 2,
        )
        splits["transfer"][skill] = neutral_instruction_skill_records(
            skill,
            "val",
            VALIDATION_LEXICON,
            transfer_pairs_per_skill,
            skill_seed + 3,
        )
    return splits


def build_expanded_dataset(
    output_dir: Union[str, Path],
    **source_options,
) -> dict:
    source_splits = expanded_source_splits(**source_options)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "expanded_resolver_version": EXPANDED_RESOLVER_VERSION,
        "family": "direct_single_span",
        "skills": list(DIRECT_SPAN_SKILLS),
        "candidate_cardinalities": [2, 3, 4],
        "splits": {},
    }
    for split in EXPANDED_SPLITS:
        records = []
        for skill in DIRECT_SPAN_SKILLS:
            lexicon = (
                VALIDATION_LEXICON
                if split in {"lexical", "transfer"}
                else TRAIN_LEXICON
            )
            records.extend(
                build_expanded_records(
                    source_splits[split][skill], split, skill, lexicon
                )
            )
        records = tuple(records)
        report = validate_expanded_records(records)
        path = output_dir / f"{split}.jsonl"
        save_expanded_records(records, path)
        report.update(
            {"path": str(path), "sha256": sha256_text(path.read_text(encoding="utf-8"))}
        )
        manifest["splits"][split] = report
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


@dataclass(frozen=True)
class EncodedExpandedExample:
    record_id: str
    input_ids: tuple[int, ...]
    sequence_tokens: int
    candidate_token_masks: tuple[tuple[bool, ...], ...]
    candidate_view_input_ids: tuple[tuple[int, ...], ...]
    candidate_view_sequence_tokens: tuple[int, ...]
    action_target: int
    candidate_target: int


def _require_control_tokens(tokenizer: ByteBPETokenizer) -> None:
    missing = set(EXPANDED_CONTROL_TOKENS) - set(tokenizer.special_token_ids)
    if missing:
        raise ValueError(
            "expanded resolver tokenizer is missing control tokens: "
            + ", ".join(sorted(missing))
        )


def _padded(tokens: list[int], block_size: int, label: str) -> tuple[int, ...]:
    if len(tokens) > block_size:
        raise ValueError(
            f"{label} needs {len(tokens)} tokens but block_size is {block_size}"
        )
    return tuple(tokens + [0] * (block_size - len(tokens)))


def encode_expanded_record(
    record: ExpandedResolverRecord,
    tokenizer: ByteBPETokenizer,
    block_size: int,
    max_candidates: int = MAX_CANDIDATES,
) -> EncodedExpandedExample:
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise TypeError("expanded typed-span resolution requires byte-BPE")
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

    masks = [
        tuple(start <= position < end for position in range(block_size))
        for start, end in spans
    ]
    while len(masks) < max_candidates:
        masks.append(tuple(False for _ in range(block_size)))

    candidate_views = []
    candidate_view_lengths = []
    for candidate in record.candidates:
        marked_prompt = record.prompt.replace(candidate.text, CANDIDATE_MARKER)
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

    return EncodedExpandedExample(
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


def encode_expanded_records(
    records: Iterable[ExpandedResolverRecord],
    tokenizer: ByteBPETokenizer,
    block_size: int,
) -> tuple[EncodedExpandedExample, ...]:
    return tuple(
        encode_expanded_record(record, tokenizer, block_size) for record in records
    )


def expanded_batch(
    examples: tuple[EncodedExpandedExample, ...],
    indices: Iterable[int],
    device: Union[str, torch.device],
) -> tuple[torch.Tensor, ...]:
    selected = tuple(examples[index] for index in indices)
    if not selected:
        raise ValueError("expanded resolver batch cannot be empty")
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
            [example.candidate_view_sequence_tokens for example in selected],
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
class ExpandedResolverOutput:
    action_logits: torch.Tensor
    candidate_logits: torch.Tensor
    loss: Optional[torch.Tensor]
    action_loss: Optional[torch.Tensor]
    candidate_loss: Optional[torch.Tensor]


class ExpandedTypedSpanResolver(nn.Module):
    """Parameter-compatible Phase 31 resolver with variable candidates."""

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
        candidate_token_masks: torch.Tensor,
        candidate_view_tokens: torch.Tensor,
        candidate_view_sequence_lengths: torch.Tensor,
        action_targets: Optional[torch.Tensor] = None,
        candidate_targets: Optional[torch.Tensor] = None,
    ) -> ExpandedResolverOutput:
        batch, candidate_count, sequence = candidate_view_tokens.shape
        combined_tokens = torch.cat(
            (tokens, candidate_view_tokens.reshape(-1, sequence)), dim=0
        )
        combined_hidden = self.hidden_states(combined_tokens)
        hidden = combined_hidden[:batch]
        candidate_hidden = combined_hidden[batch:].reshape(
            batch, candidate_count, sequence, -1
        )
        positions = sequence_lengths - 1
        queries = hidden[torch.arange(batch, device=tokens.device), positions]
        action_logits = self.action_head(queries)

        candidate_positions = candidate_view_sequence_lengths - 1
        batch_indices = torch.arange(batch, device=tokens.device).unsqueeze(1)
        candidate_indices = torch.arange(
            candidate_count, device=tokens.device
        ).unsqueeze(0)
        candidate_states = candidate_hidden[
            batch_indices, candidate_indices, candidate_positions
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
                    candidate_logits[resolve_rows], candidate_targets[resolve_rows]
                )
                loss = candidate_loss if loss is None else loss + candidate_loss
        return ExpandedResolverOutput(
            action_logits=action_logits,
            candidate_logits=candidate_logits,
            loss=loss,
            action_loss=action_loss,
            candidate_loss=candidate_loss,
        )


@dataclass(frozen=True)
class ExpandedResolverDecision:
    action: str
    text: Optional[str]
    selected_candidate_index: Optional[int]
    selected_value: Optional[str]
    guarded: bool = False


def realize_expanded_decision(
    record: ExpandedResolverRecord,
    action: str,
    candidate_index: Optional[int],
) -> ExpandedResolverDecision:
    if action not in RESOLVER_ACTIONS:
        raise ValueError(f"unknown resolver action: {action}")
    if action == GENERATE_ACTION:
        return ExpandedResolverDecision(action, None, None, None)
    if action == CLARIFY_ACTION:
        return ExpandedResolverDecision(action, CLARIFICATION_RESPONSE, None, None)
    valid = (
        candidate_index is not None
        and 0 <= candidate_index < len(record.candidates)
        and record.expected_type is not None
        and record.candidates[candidate_index].value_type == record.expected_type
        and record.response_template is not None
        and record.response_template.count(RESOLVED_VALUE_MARKER) == 1
    )
    if not valid:
        return ExpandedResolverDecision(
            CLARIFY_ACTION, CLARIFICATION_RESPONSE, None, None, guarded=True
        )
    assert candidate_index is not None
    assert record.response_template is not None
    candidate = record.candidates[candidate_index]
    return ExpandedResolverDecision(
        RESOLVE_ACTION,
        record.response_template.replace(RESOLVED_VALUE_MARKER, candidate.text),
        candidate_index,
        candidate.text,
    )


def _flat_prediction_summary(rows: tuple[dict, ...]) -> dict:
    def subset_rate(subset, key: str) -> float:
        if not subset:
            return 0.0
        return sum(bool(row[key]) for row in subset) / len(subset)

    resolve_rows = tuple(row for row in rows if row["expected_action"] == RESOLVE_ACTION)
    clarify_rows = tuple(row for row in rows if row["expected_action"] == CLARIFY_ACTION)
    generate_rows = tuple(row for row in rows if row["expected_action"] == GENERATE_ACTION)
    conversations = defaultdict(list)
    for row in resolve_rows:
        conversations[row["conversation_id"]].append(row)
    return {
        "examples": len(rows),
        "action_accuracy": subset_rate(rows, "action_correct"),
        "resolve_examples": len(resolve_rows),
        "resolve_action_accuracy": subset_rate(resolve_rows, "action_correct"),
        "end_to_end_resolve_accuracy": subset_rate(resolve_rows, "end_to_end_correct"),
        "counterfactual_pair_resolve_accuracy": (
            sum(
                len(pair) == 2 and all(row["end_to_end_correct"] for row in pair)
                for pair in conversations.values()
            )
            / len(conversations)
            if conversations
            else 0.0
        ),
        "exact_realization_rate": subset_rate(resolve_rows, "exact_realization"),
        "value_missing_rate": subset_rate(resolve_rows, "value_missing"),
        "wrong_alternative_rate": subset_rate(resolve_rows, "wrong_alternative"),
        "clarify_examples": len(clarify_rows),
        "clarify_accuracy": subset_rate(clarify_rows, "action_correct"),
        "generate_examples": len(generate_rows),
        "generate_accuracy": subset_rate(generate_rows, "action_correct"),
        "guarded_rate": subset_rate(rows, "guarded"),
    }


def summarize_expanded_predictions(rows: Iterable[dict]) -> dict:
    rows = tuple(rows)
    if not rows:
        raise ValueError("resolver prediction rows cannot be empty")
    summary = _flat_prediction_summary(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["skill"]].append(row)
    summary["per_skill"] = {
        skill: _flat_prediction_summary(tuple(skill_rows))
        for skill, skill_rows in sorted(grouped.items())
    }
    resolve_rows = tuple(
        row for row in rows if row["expected_action"] == RESOLVE_ACTION
    )
    by_width = defaultdict(list)
    by_skill_width = defaultdict(lambda: defaultdict(list))
    for row in resolve_rows:
        width = str(row["candidate_count"])
        by_width[width].append(row)
        by_skill_width[row["skill"]][width].append(row)
    summary["per_candidate_width"] = {
        width: _flat_prediction_summary(tuple(width_rows))
        for width, width_rows in sorted(by_width.items())
    }
    summary["per_skill_candidate_width"] = {
        skill: {
            width: _flat_prediction_summary(tuple(width_rows))
            for width, width_rows in sorted(width_groups.items())
        }
        for skill, width_groups in sorted(by_skill_width.items())
    }
    summary["cases"] = dict(sorted(Counter(row["case"] for row in rows).items()))
    summary["skills"] = dict(sorted(Counter(row["skill"] for row in rows).items()))
    return summary
