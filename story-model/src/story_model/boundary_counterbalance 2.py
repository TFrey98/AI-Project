"""Phase 35 compositional boundary counterbalance data.

The Phase 34k intervention localized the residual boundary error to
``proposal_key(h_current_token)``.  This module constructs a training-only
counterbalance in which the same BPE token is observed at the beginning and
inside of matched route spans.  Separate focus-token identities are reserved
for lexical and combined-transfer evaluation.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Union

from story_model.data import ByteBPETokenizer
from story_model.expanded_typed_span_resolver import (
    EXPANDED_SPLITS,
    RESOLVE_ACTION,
    ExpandedCandidate,
    ExpandedResolverRecord,
    load_expanded_records,
    save_expanded_records,
)
from story_model.explicit_offset_candidate_proposer import (
    OUTSIDE_TAG,
    begin_tag,
    encode_proposal_records,
    inside_tag,
)
from story_model.provenance import canonical_json_sha256


BOUNDARY_COUNTERBALANCE_VERSION = 2
EXPECTED_PHASE34K_VERSION = 1
EXPECTED_PHASE34K_BRANCH = "local_token_state_dominant"
DEFAULT_TRAIN_FOCUS_TOKENS = 8
DEFAULT_HELDOUT_FOCUS_TOKENS = 4
FOCUS_TOKEN_WIDTH = 2
FOCUS_SPLIT_KIND = {
    "train": "train",
    "val": "train",
    "lexical": "heldout",
    "paraphrase": "train",
    "transfer": "heldout",
}

_SPLIT_MARKERS = {
    "train": "ember",
    "val": "willow",
    "lexical": "quartz",
    "paraphrase": "harbor",
    "transfer": "silver",
}

_ROUTE_FORMS = (
    ("causeway", "upper", "passage"),
    ("bridge", "lower", "arcade"),
    ("lane", "north", "crossing"),
    ("track", "south", "stair"),
    ("gate", "river", "walkway"),
    ("path", "stone", "tunnel"),
    ("road", "garden", "corridor"),
    ("ramp", "market", "gallery"),
    ("trail", "western", "door"),
    ("route", "eastern", "footbridge"),
    ("way", "cliff", "aqueduct"),
    ("turn", "forest", "causeway"),
)

_GENERATED_ROUTE_FILLER = tuple(
    sorted(
        set(_SPLIT_MARKERS.values())
        | {fragment for form in _ROUTE_FORMS for fragment in form}
    )
)


class FocusTokenCollisionError(ValueError):
    """Raised when a focus identity leaks into another generated route."""


def _sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_phase34k_premise(summary: dict) -> dict:
    """Validate that Phase 34k selected a data-only local-state test."""

    failures = []
    if summary.get("context_path_intervention_version") != (
        EXPECTED_PHASE34K_VERSION
    ):
        failures.append("Phase 34k summary has the wrong audit version")
    decision = summary.get("decision", {})
    if decision.get("branch") != EXPECTED_PHASE34K_BRANCH:
        failures.append(
            "Phase 34k did not select local-token-state dominance"
        )
    for field in (
        "training_authorized",
        "full_phase34_evaluation_authorized",
        "checkpoint_promotion_authorized",
    ):
        if decision.get(field) is not False:
            failures.append(f"Phase 34k has invalid {field}")

    focus_ids = summary.get("focus_token_ids", {})
    if set(focus_ids) != {"or", "ro"} or any(
        type(value) is not int for value in focus_ids.values()
    ):
        failures.append("Phase 34k focus token IDs are missing or invalid")
    elif len(set(focus_ids.values())) != 2:
        failures.append("Phase 34k focus token IDs are not distinct")

    tokenizer_hashes = {
        model.get("checkpoint", {}).get("tokenizer_sha256")
        for model in summary.get("models", {}).values()
    }
    tokenizer_hashes.discard(None)
    if len(tokenizer_hashes) != 1:
        failures.append("Phase 34k does not identify one tokenizer")

    if failures:
        raise ValueError("; ".join(failures))
    return {
        "branch": decision["branch"],
        "focus_token_ids": dict(focus_ids),
        "tokenizer_sha256": next(iter(tokenizer_hashes)),
    }


def candidate_focus_tokens(
    tokenizer: ByteBPETokenizer,
    excluded_token_ids: Iterable[int] = (),
) -> tuple[tuple[int, str], ...]:
    """Return printable two-byte BPE identities eligible for balancing."""

    excluded = {int(token_id) for token_id in excluded_token_ids}
    special_ids = set(tokenizer.special_token_ids.values())
    candidates = []
    for token_id in range(tokenizer.vocab_size):
        if token_id in excluded or token_id in special_ids:
            continue
        raw = tokenizer.token_bytes(token_id)
        if len(raw) != FOCUS_TOKEN_WIDTH:
            continue
        try:
            text = raw.decode("ascii")
        except UnicodeDecodeError:
            continue
        if (
            text.isalpha()
            and text.islower()
            and not any(text in fragment for fragment in _GENERATED_ROUTE_FILLER)
        ):
            candidates.append((token_id, text))
    return tuple(candidates)


def scene_route_pairs(
    records: Iterable[ExpandedResolverRecord],
) -> tuple[tuple[ExpandedResolverRecord, ExpandedResolverRecord], ...]:
    """Recover complete Phase 31 supported scene-route pairs in file order."""

    grouped = defaultdict(list)
    order = []
    for record in records:
        if not (
            record.skill == "scene_route"
            and record.case == "supported"
            and record.expected_action == RESOLVE_ACTION
        ):
            continue
        if record.conversation_id not in grouped:
            order.append(record.conversation_id)
        grouped[record.conversation_id].append(record)

    pairs = []
    for conversation_id in order:
        pair = tuple(grouped[conversation_id])
        if len(pair) != 2:
            raise ValueError(
                f"scene-route conversation {conversation_id!r} is not a pair"
            )
        if pair[0].candidates != pair[1].candidates:
            raise ValueError("scene-route pair candidate inventory changed")
        if len(pair[0].candidates) != 2:
            raise ValueError("Phase 35 requires two-candidate Phase 31 pairs")
        if pair[0].selected_candidate_index == pair[1].selected_candidate_index:
            raise ValueError("scene-route pair target does not flip")
        pairs.append((pair[0], pair[1]))
    if not pairs:
        raise ValueError("no supported Phase 31 scene-route pairs found")
    return tuple(pairs)


def _replace_candidates(prompt: str, replacements: dict[str, str]) -> str:
    placeholders = {}
    result = prompt
    for index, old in enumerate(sorted(replacements, key=len, reverse=True)):
        placeholder = f"__PHASE35_ROUTE_{index}__"
        if old not in result:
            raise ValueError(f"source prompt does not contain route {old!r}")
        result = result.replace(old, placeholder)
        placeholders[placeholder] = replacements[old]
    for placeholder, new in placeholders.items():
        result = result.replace(placeholder, new)
    if any(old in result for old in replacements):
        raise RuntimeError("source route replacement was incomplete")
    return result


def _route_values(token_text: str, split: str, variant: int) -> tuple[str, str]:
    first = _ROUTE_FORMS[variant % len(_ROUTE_FORMS)]
    second = _ROUTE_FORMS[(variant * 5 + 3) % len(_ROUTE_FORMS)]
    marker = _SPLIT_MARKERS[split]
    start_value = f"{token_text}-{first[0]}-{marker}"
    inside_value = f"{second[1]}-{token_text}-{second[2]}-{marker}"
    if start_value in inside_value or inside_value in start_value:
        raise RuntimeError("counterbalance route values overlap")
    return start_value, inside_value


def _transform_pair(
    pair: tuple[ExpandedResolverRecord, ExpandedResolverRecord],
    split: str,
    pair_index: int,
    token_id: int,
    token_text: str,
    variant: int,
) -> tuple[ExpandedResolverRecord, ExpandedResolverRecord]:
    old_values = tuple(candidate.text for candidate in pair[0].candidates)
    new_values = _route_values(token_text, split, variant)
    replacements = dict(zip(old_values, new_values))
    candidates = tuple(
        ExpandedCandidate(value, "route") for value in new_values
    )
    transformed = []
    for side, source in enumerate(pair):
        selected = source.selected_candidate_index
        if selected is None:
            raise ValueError("supported source pair has no selected candidate")
        alternative = 1 - selected
        transformed.append(
            replace(
                source,
                record_id=f"phase35_{split}_{pair_index:05d}:{side}",
                source_context_id=(
                    f"phase35_{split}_{pair_index:05d}_token_{token_id}:{side}"
                ),
                conversation_id=f"phase35_{split}_{pair_index:05d}",
                split=split,
                prompt=_replace_candidates(source.prompt, replacements),
                candidates=candidates,
                expected_value=new_values[selected],
                alternative_value=new_values[alternative],
                source_phase="phase35",
            )
        )
    return transformed[0], transformed[1]


def focus_boundary_counts(
    records: Iterable[ExpandedResolverRecord],
    tokenizer: ByteBPETokenizer,
    block_size: int,
    token_id: int,
) -> dict[str, int]:
    """Count B/I/O labels at byte offset zero for one exact BPE token."""

    counts = Counter()
    examples = encode_proposal_records(tuple(records), tokenizer, block_size)
    route_begin = begin_tag("route")
    route_inside = inside_tag("route")
    for example in examples:
        byte_cursor = 0
        for position in range(example.prompt_token_count):
            current_id = example.input_ids[position]
            first_tag = example.prompt_byte_tags[byte_cursor]
            if current_id == token_id:
                if first_tag == route_begin:
                    counts["B"] += 1
                elif first_tag == route_inside:
                    counts["I"] += 1
                else:
                    counts["O"] += 1
            byte_cursor += tokenizer.token_byte_length(current_id)
        if byte_cursor != len(example.prompt_byte_tags):
            raise RuntimeError("focus-token labels did not align to prompt bytes")
    return {label: int(counts[label]) for label in ("B", "I", "O")}


def _validated_transform(
    pair: tuple[ExpandedResolverRecord, ExpandedResolverRecord],
    split: str,
    pair_index: int,
    token_id: int,
    token_text: str,
    tokenizer: ByteBPETokenizer,
    block_size: int,
    variant_seed: int,
) -> tuple[tuple[ExpandedResolverRecord, ExpandedResolverRecord], dict]:
    for offset in range(len(_ROUTE_FORMS) * 2):
        variant = variant_seed + offset
        transformed = _transform_pair(
            pair,
            split,
            pair_index,
            token_id,
            token_text,
            variant,
        )
        try:
            counts = focus_boundary_counts(
                transformed, tokenizer, block_size, token_id
            )
        except (RuntimeError, ValueError):
            continue
        if counts["B"] > 0 and counts["B"] == counts["I"]:
            return transformed, counts
    raise ValueError(
        f"token {token_id} ({token_text!r}) cannot form a balanced {split} pair"
    )


def _token_viable(
    token: tuple[int, str],
    representative_pairs: dict[str, tuple[ExpandedResolverRecord, ExpandedResolverRecord]],
    required_splits: tuple[str, ...],
    tokenizer: ByteBPETokenizer,
    block_size: int,
) -> bool:
    token_id, token_text = token
    for index, split in enumerate(required_splits):
        try:
            _validated_transform(
                representative_pairs[split],
                split,
                0,
                token_id,
                token_text,
                tokenizer,
                block_size,
                token_id + index,
            )
        except ValueError:
            return False
    return True


def select_focus_token_pools(
    tokenizer: ByteBPETokenizer,
    source_pairs: dict[str, tuple[tuple[ExpandedResolverRecord, ExpandedResolverRecord], ...]],
    excluded_token_ids: Iterable[int],
    train_count: int = DEFAULT_TRAIN_FOCUS_TOKENS,
    heldout_count: int = DEFAULT_HELDOUT_FOCUS_TOKENS,
    block_size: int = 1024,
    seed: int = 1337,
) -> dict[str, tuple[tuple[int, str], ...]]:
    """Select disjoint focus identities that work in the frozen templates."""

    if train_count < 1 or heldout_count < 1:
        raise ValueError("focus-token pool sizes must be positive")
    representative = {split: pairs[0] for split, pairs in source_pairs.items()}
    candidates = list(candidate_focus_tokens(tokenizer, excluded_token_ids))
    random.Random(seed).shuffle(candidates)
    train = []
    heldout = []
    for token in candidates:
        if len(train) < train_count and _token_viable(
            token,
            representative,
            ("train", "val", "paraphrase"),
            tokenizer,
            block_size,
        ):
            train.append(token)
            continue
        if len(heldout) < heldout_count and _token_viable(
            token,
            representative,
            ("lexical", "transfer"),
            tokenizer,
            block_size,
        ):
            heldout.append(token)
        if len(train) == train_count and len(heldout) == heldout_count:
            break
    if len(train) != train_count or len(heldout) != heldout_count:
        raise ValueError(
            "not enough viable two-byte BPE identities for disjoint "
            f"counterbalance pools: train {len(train)}/{train_count}, "
            f"heldout {len(heldout)}/{heldout_count}"
        )
    return {"train": tuple(train), "heldout": tuple(heldout)}


def build_counterbalance_records(
    source_pairs: dict[str, tuple[tuple[ExpandedResolverRecord, ExpandedResolverRecord], ...]],
    focus_pools: dict[str, tuple[tuple[int, str], ...]],
    tokenizer: ByteBPETokenizer,
    block_size: int,
) -> tuple[dict[str, tuple[ExpandedResolverRecord, ...]], dict]:
    """Transform every Phase 31 scene-route pair and prove B/I balance."""

    built = {}
    reports = {}
    for split in EXPANDED_SPLITS:
        pool_kind = FOCUS_SPLIT_KIND[split]
        pool = focus_pools[pool_kind]
        if not pool:
            raise ValueError(f"Phase 35 {pool_kind} focus pool is empty")
        output = []
        assigned_counts = defaultdict(Counter)
        for pair_index, pair in enumerate(source_pairs[split]):
            built_pair = None
            for rotation in range(len(pool)):
                token_id, token_text = pool[(pair_index + rotation) % len(pool)]
                try:
                    built_pair, counts = _validated_transform(
                        pair,
                        split,
                        pair_index,
                        token_id,
                        token_text,
                        tokenizer,
                        block_size,
                        pair_index,
                    )
                except ValueError:
                    continue
                assigned_counts[token_id].update(counts)
                break
            if built_pair is None:
                raise ValueError(
                    f"no {pool_kind} focus token works for {split} pair "
                    f"{pair_index}"
                )
            output.extend(built_pair)
        records = tuple(output)
        expected_ids = {token_id for token_id, _ in pool}
        observed_ids = set(assigned_counts)
        if observed_ids != expected_ids:
            missing = sorted(expected_ids - observed_ids)
            raise ValueError(f"Phase 35 {split} omitted focus tokens {missing}")
        global_counts = {}
        for token_id, counts in assigned_counts.items():
            actual = focus_boundary_counts(
                records, tokenizer, block_size, token_id
            )
            if (
                actual["B"] != counts["B"]
                or actual["I"] != counts["I"]
            ):
                raise FocusTokenCollisionError(
                    f"Phase 35 {split} token {token_id} occurs as B/I in "
                    "another focus token's generated route"
                )
            if actual["B"] < 1 or actual["B"] != actual["I"]:
                raise ValueError(
                    f"Phase 35 {split} token {token_id} is not B/I balanced"
                )
            global_counts[token_id] = actual
        built[split] = records
        reports[split] = {
            "rows": len(records),
            "pairs": len(records) // 2,
            "focus_pool": pool_kind,
            "focus_label_counts": {
                str(token_id): {
                    label: int(counts[label]) for label in ("B", "I", "O")
                }
                for token_id, counts in sorted(global_counts.items())
            },
            "assigned_focus_label_counts": {
                str(token_id): {
                    label: int(counts[label]) for label in ("B", "I", "O")
                }
                for token_id, counts in sorted(assigned_counts.items())
            },
        }
    return built, reports


def candidate_value_token_ids(
    records: Iterable[ExpandedResolverRecord],
    tokenizer: ByteBPETokenizer,
    block_size: int = 1024,
) -> set[int]:
    """Return token identities used inside tokenized gold candidate spans."""

    token_ids = set()
    examples = encode_proposal_records(records, tokenizer, block_size)
    for example in examples:
        byte_cursor = 0
        for position in range(example.prompt_token_count):
            token_id = example.input_ids[position]
            width = tokenizer.token_byte_length(token_id)
            tags = example.prompt_byte_tags[byte_cursor : byte_cursor + width]
            if any(tag != OUTSIDE_TAG for tag in tags):
                token_ids.add(token_id)
            byte_cursor += width
    return token_ids


def build_boundary_counterbalance_dataset(
    source_records: dict[str, tuple[ExpandedResolverRecord, ...]],
    tokenizer: ByteBPETokenizer,
    phase34k_summary: dict,
    output_dir: Union[str, Path],
    source_files: dict[str, Union[str, Path]],
    phase34k_summary_path: Union[str, Path],
    train_focus_tokens: int = DEFAULT_TRAIN_FOCUS_TOKENS,
    heldout_focus_tokens: int = DEFAULT_HELDOUT_FOCUS_TOKENS,
    block_size: int = 1024,
    seed: int = 1337,
) -> dict:
    """Build and write the complete tokenizer-aware Phase 35 dataset."""

    premise = validate_phase34k_premise(phase34k_summary)
    tokenizer_sha256 = canonical_json_sha256(tokenizer.to_dict())
    if tokenizer_sha256 != premise["tokenizer_sha256"]:
        raise ValueError("Phase 33c tokenizer does not match Phase 34k")
    for split in ("lexical", "transfer"):
        prior = phase34k_summary.get("data_files", {}).get(split, {})
        current_path = Path(source_files[split])
        if prior.get("sha256") != _sha256(current_path):
            raise ValueError(f"Phase 31 {split} data changed after Phase 34k")
        if int(prior.get("rows", -1)) != len(source_records[split]):
            raise ValueError(f"Phase 31 {split} row count changed after Phase 34k")

    pairs = {
        split: scene_route_pairs(source_records[split])
        for split in EXPANDED_SPLITS
    }
    pools = None
    records = None
    reports = None
    for attempt in range(64):
        candidate_pools = select_focus_token_pools(
            tokenizer,
            pairs,
            premise["focus_token_ids"].values(),
            train_count=train_focus_tokens,
            heldout_count=heldout_focus_tokens,
            block_size=block_size,
            seed=seed + attempt * 104729,
        )
        try:
            candidate_records, candidate_reports = build_counterbalance_records(
                pairs, candidate_pools, tokenizer, block_size
            )
        except FocusTokenCollisionError:
            continue
        heldout_ids = {
            token_id for token_id, _ in candidate_pools["heldout"]
        }
        training_value_ids = candidate_value_token_ids(
            candidate_records["train"], tokenizer, block_size
        )
        if not heldout_ids & training_value_ids:
            pools = candidate_pools
            records = candidate_records
            reports = candidate_reports
            break
    if pools is None or records is None or reports is None:
        raise ValueError(
            "could not construct heldout focus identities absent from all "
            "counterbalance training values"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in EXPANDED_SPLITS:
        path = output_dir / f"{split}.jsonl"
        save_expanded_records(records[split], path)
        reports[split]["sha256"] = _sha256(path)

    manifest = {
        "boundary_counterbalance_version": BOUNDARY_COUNTERBALANCE_VERSION,
        "seed": seed,
        "block_size": block_size,
        "strategy": (
            "same local BPE identity receives balanced route B/I labels; "
            "lexical and transfer focus identities do not occur in any "
            "counterbalance training route value"
        ),
        "focus_collision_policy": {
            "generated_route_filler": list(_GENERATED_ROUTE_FILLER),
            "generated_route_filler_sha256": canonical_json_sha256(
                list(_GENERATED_ROUTE_FILLER)
            ),
            "focus_bytes_absent_from_generated_route_filler": True,
            "global_positive_counts_must_equal_assigned_counts": True,
            "outside_counts_are_measured_not_balanced": True,
        },
        "phase34k_premise": {
            **premise,
            "path": str(phase34k_summary_path),
            "sha256": _sha256(phase34k_summary_path),
        },
        "tokenizer_sha256": tokenizer_sha256,
        "source_phase31_files": {
            split: {
                "path": str(source_files[split]),
                "sha256": _sha256(source_files[split]),
                "rows": len(source_records[split]),
            }
            for split in EXPANDED_SPLITS
        },
        "focus_tokens": {
            kind: [
                {
                    "token_id": token_id,
                    "text": text,
                    "bytes_hex": tokenizer.token_bytes(token_id).hex(),
                }
                for token_id, text in pool
            ]
            for kind, pool in pools.items()
        },
        "value_token_disjointness": {
            "heldout_focus_ids_in_training_values": [],
            "training_value_token_count": len(
                candidate_value_token_ids(records["train"], tokenizer, block_size)
            ),
        },
        "splits": reports,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_counterbalance_manifest(manifest, output_dir, tokenizer)
    return manifest


def validate_counterbalance_manifest(
    manifest: dict,
    data_dir: Union[str, Path],
    tokenizer: ByteBPETokenizer,
) -> dict[str, tuple[int, ...]]:
    """Validate saved data hashes and return the two focus-token pools."""

    failures = []
    if manifest.get("boundary_counterbalance_version") != (
        BOUNDARY_COUNTERBALANCE_VERSION
    ):
        failures.append("counterbalance manifest has the wrong version")
    tokenizer_sha256 = canonical_json_sha256(tokenizer.to_dict())
    if manifest.get("tokenizer_sha256") != tokenizer_sha256:
        failures.append("counterbalance tokenizer does not match checkpoint")

    pools = {}
    for kind in ("train", "heldout"):
        entries = manifest.get("focus_tokens", {}).get(kind, ())
        ids = tuple(entry.get("token_id") for entry in entries)
        if not ids or any(type(token_id) is not int for token_id in ids):
            failures.append(f"counterbalance {kind} focus pool is invalid")
            ids = ()
        if len(set(ids)) != len(ids):
            failures.append(f"counterbalance {kind} focus pool repeats IDs")
        for entry in entries:
            token_id = entry.get("token_id")
            if type(token_id) is not int or not 0 <= token_id < tokenizer.vocab_size:
                continue
            raw = tokenizer.token_bytes(token_id)
            if entry.get("bytes_hex") != raw.hex():
                failures.append(
                    f"counterbalance {kind} token {token_id} bytes changed"
                )
            try:
                text = raw.decode("ascii")
            except UnicodeDecodeError:
                text = None
            if entry.get("text") != text:
                failures.append(
                    f"counterbalance {kind} token {token_id} text changed"
                )
            if text is not None and any(
                text in fragment for fragment in _GENERATED_ROUTE_FILLER
            ):
                failures.append(
                    f"counterbalance {kind} token {token_id} collides with "
                    "generated route filler"
                )
        pools[kind] = ids
    if set(pools.get("train", ())) & set(pools.get("heldout", ())):
        failures.append("train and heldout focus-token pools overlap")
    excluded = set(
        manifest.get("phase34k_premise", {})
        .get("focus_token_ids", {})
        .values()
    )
    if excluded & (set(pools.get("train", ())) | set(pools.get("heldout", ()))):
        failures.append("Phase 35 reuses a Phase 34k focus identity")
    premise = manifest.get("phase34k_premise", {})
    if premise.get("branch") != EXPECTED_PHASE34K_BRANCH:
        failures.append("counterbalance manifest has the wrong Phase 34k premise")
    if premise.get("tokenizer_sha256") != tokenizer_sha256:
        failures.append("Phase 34k premise tokenizer does not match checkpoint")
    collision_policy = manifest.get("focus_collision_policy", {})
    if collision_policy.get("generated_route_filler_sha256") != (
        canonical_json_sha256(list(_GENERATED_ROUTE_FILLER))
    ):
        failures.append("counterbalance generated route filler changed")
    for field in (
        "focus_bytes_absent_from_generated_route_filler",
        "global_positive_counts_must_equal_assigned_counts",
        "outside_counts_are_measured_not_balanced",
    ):
        if collision_policy.get(field) is not True:
            failures.append(f"counterbalance collision policy lost {field}")

    data_dir = Path(data_dir)
    loaded = {}
    block_size = int(manifest.get("block_size", 0))
    if block_size < 1:
        failures.append("counterbalance manifest has an invalid block size")
        block_size = 1024
    for split in EXPANDED_SPLITS:
        path = data_dir / f"{split}.jsonl"
        report = manifest.get("splits", {}).get(split, {})
        if not path.is_file() or report.get("sha256") != _sha256(path):
            failures.append(f"counterbalance {split} data hash changed")
            continue
        loaded[split] = load_expanded_records(path)
        if int(report.get("rows", -1)) != len(loaded[split]):
            failures.append(f"counterbalance {split} row count changed")
        expected_kind = FOCUS_SPLIT_KIND[split]
        if report.get("focus_pool") != expected_kind:
            failures.append(f"counterbalance {split} has the wrong focus pool")
        expected_ids = {str(token_id) for token_id in pools.get(expected_kind, ())}
        counts = report.get("focus_label_counts", {})
        assigned_counts = report.get("assigned_focus_label_counts", {})
        if set(counts) != expected_ids:
            failures.append(f"counterbalance {split} focus support changed")
        if set(assigned_counts) != expected_ids:
            failures.append(
                f"counterbalance {split} assigned focus support changed"
            )
        for token_id, labels in counts.items():
            begin = int(labels.get("B", 0))
            inside = int(labels.get("I", 0))
            if begin < 1 or begin != inside:
                failures.append(
                    f"counterbalance {split} token {token_id} is not B/I balanced"
                )
        for token_id in pools.get(expected_kind, ()):
            actual = focus_boundary_counts(
                loaded[split], tokenizer, block_size, token_id
            )
            declared = counts.get(str(token_id))
            if declared != actual:
                failures.append(
                    f"counterbalance {split} token {token_id} labels changed"
                )
            assigned = assigned_counts.get(str(token_id), {})
            if (
                int(assigned.get("B", -1)) != actual["B"]
                or int(assigned.get("I", -1)) != actual["I"]
            ):
                failures.append(
                    f"counterbalance {split} token {token_id} has an "
                    "incidental positive-label collision"
                )
    if "train" in loaded:
        training_value_ids = candidate_value_token_ids(
            loaded["train"],
            tokenizer,
            block_size,
        )
        overlap = sorted(set(pools.get("heldout", ())) & training_value_ids)
        if overlap:
            failures.append(
                "heldout focus IDs occur in counterbalance training values: "
                + ", ".join(str(token_id) for token_id in overlap)
            )
        declared = manifest.get("value_token_disjointness", {}).get(
            "heldout_focus_ids_in_training_values"
        )
        if declared != []:
            failures.append("counterbalance manifest does not declare zero overlap")
        declared_count = manifest.get("value_token_disjointness", {}).get(
            "training_value_token_count"
        )
        if declared_count != len(training_value_ids):
            failures.append("counterbalance training value-token count changed")
    if failures:
        raise ValueError("; ".join(failures))
    return pools
