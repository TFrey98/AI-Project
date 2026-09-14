"""Phase 36b paired identity-invariance training utilities.

The intervention changes one token identity inside a gold route span while
holding sequence length, byte labels, and every other token fixed. It is an
in-loop augmentation: no encoded dataset row is mutated or written to disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.nn import functional as F

from story_model.data import ByteBPETokenizer
from story_model.explicit_offset_candidate_proposer import begin_tag, inside_tag
from story_model.provenance import canonical_json_sha256


PAIRED_IDENTITY_INVARIANCE_VERSION = 2
IDENTITY_SWAP_POOL_SCHEMA_VERSION = 1
BEGIN_ONLY_POSITION_MODE = "begin_only"
BEGIN_AND_INSIDE_POSITION_MODE = "begin_and_inside"


@dataclass(frozen=True)
class IdentitySwap:
    original_batch_index: int
    swapped_batch_index: int
    token_position: int
    byte_offset: int
    original_token_id: int
    replacement_token_id: int
    token_width: int
    gold_tag: int


def registered_identity_ids(counterbalance_manifest: dict) -> tuple[int, ...]:
    """Return the registered Phase 35 train/held-out and legacy identities."""
    values = []
    for group, expected in (("train", 8), ("heldout", 4)):
        entries = counterbalance_manifest.get("focus_tokens", {}).get(group, ())
        group_ids = [int(entry["token_id"]) for entry in entries]
        if len(group_ids) != expected or len(set(group_ids)) != expected:
            raise ValueError(
                f"identity-invariance requires {expected} distinct {group} IDs"
            )
        values.extend(group_ids)
    legacy = counterbalance_manifest.get("phase34k_premise", {}).get(
        "focus_token_ids", {}
    )
    if set(legacy) != {"or", "ro"}:
        raise ValueError("identity-invariance requires legacy or/ro identities")
    values.extend(int(token_id) for token_id in legacy.values())
    if len(values) != 14 or len(set(values)) != 14:
        raise ValueError("the 14 registered identities must be distinct")
    return tuple(sorted(values))


def build_identity_swap_pool(
    tokenizer: ByteBPETokenizer,
    excluded_token_ids: Iterable[int],
) -> dict:
    """Build the deterministic, width-stratified replacement-token pool."""
    excluded = {int(token_id) for token_id in excluded_token_ids}
    special = set(tokenizer.special_token_ids.values())
    if excluded & special:
        raise ValueError("registered identities cannot be special tokens")
    by_width: dict[int, list[int]] = {}
    for token_id in range(tokenizer.vocab_size):
        if token_id in excluded or token_id in special:
            continue
        width = tokenizer.token_byte_length(token_id)
        if width < 1:
            raise ValueError(f"token {token_id} has no byte representation")
        by_width.setdefault(width, []).append(token_id)
    if not by_width or not all(by_width.values()):
        raise ValueError("identity-invariance swap pool is empty")
    return {
        "paired_identity_invariance_version": IDENTITY_SWAP_POOL_SCHEMA_VERSION,
        "strategy": (
            "all non-special tokenizer identities grouped by exact byte width, "
            "excluding the 14 Phase 35/36a registered identities"
        ),
        "tokenizer_sha256": canonical_json_sha256(tokenizer.to_dict()),
        "excluded_registered_token_ids": sorted(excluded),
        "excluded_special_token_ids": sorted(special),
        "eligible_token_ids_by_width": {
            str(width): token_ids for width, token_ids in sorted(by_width.items())
        },
        "eligible_token_count": sum(len(values) for values in by_width.values()),
    }


def validate_identity_swap_pool(
    pool: dict,
    tokenizer: ByteBPETokenizer,
    excluded_token_ids: Iterable[int],
) -> dict[int, tuple[int, ...]]:
    expected = build_identity_swap_pool(tokenizer, excluded_token_ids)
    if pool != expected:
        raise ValueError("checkpoint identity-invariance swap pool changed")
    return {
        int(width): tuple(int(token_id) for token_id in token_ids)
        for width, token_ids in pool["eligible_token_ids_by_width"].items()
    }


def apply_same_width_swap(
    tokens: torch.Tensor,
    batch_index: int,
    token_position: int,
    replacement_token_id: int,
    tokenizer: ByteBPETokenizer,
) -> torch.Tensor:
    original_token_id = int(tokens[batch_index, token_position])
    original_width = tokenizer.token_byte_length(original_token_id)
    replacement_width = tokenizer.token_byte_length(replacement_token_id)
    if replacement_width != original_width:
        raise ValueError(
            f"identity swap width mismatch: {original_width} != "
            f"{replacement_width}"
        )
    swapped = tokens[batch_index].clone()
    swapped[token_position] = int(replacement_token_id)
    changed = torch.nonzero(swapped != tokens[batch_index], as_tuple=False).flatten()
    if changed.tolist() != [token_position]:
        raise RuntimeError("identity swap must change exactly one token position")
    return swapped


def eligible_swap_candidates(
    tokens: torch.Tensor,
    prompt_token_count: int,
    tag_targets: torch.Tensor,
    tokenizer: ByteBPETokenizer,
    pool_by_width: dict[int, tuple[int, ...]],
    begin_only: bool,
) -> tuple[tuple[int, int, int, int, tuple[int, ...]], ...]:
    """Return every eligible ``(position, gold_tag, token_id, width, replacements)``.

    ``tokens`` and ``tag_targets`` are already sliced to one row. This is the
    single source of truth for swap eligibility, shared by
    ``paired_identity_invariance_batch`` and by audits that must count
    eligible rows without performing the random pairing itself (e.g. proving
    the begin-only restriction did not shrink pairing opportunity).
    """
    route_tags = (
        {begin_tag("route")}
        if begin_only
        else {begin_tag("route"), inside_tag("route")}
    )
    candidates = []
    for token_position in range(prompt_token_count):
        gold_tag = int(tag_targets[token_position, 0])
        if gold_tag not in route_tags:
            continue
        original_token_id = int(tokens[token_position])
        width = tokenizer.token_byte_length(original_token_id)
        replacements = tuple(
            token_id
            for token_id in pool_by_width.get(width, ())
            if token_id != original_token_id
        )
        if replacements:
            candidates.append(
                (token_position, gold_tag, original_token_id, width, replacements)
            )
    return tuple(candidates)


def paired_identity_invariance_batch(
    batch: tuple[torch.Tensor, ...],
    tokenizer: ByteBPETokenizer,
    pool_by_width: dict[int, tuple[int, ...]],
    begin_only: bool = False,
) -> tuple[tuple[torch.Tensor, ...], tuple[IdentitySwap, ...]]:
    """Append one independently supervised swap for every route-bearing row.

    When ``begin_only`` is set (Phase 36c), only gold ``B:route`` positions
    are eligible for pairing; ``I:route`` positions are never selected. This
    is the sole behavioral difference from Phase 36b's begin-and-inside
    pairing.
    """
    tokens, sequence_lengths, prompt_token_counts, tag_targets = batch
    swapped_rows, swapped_lengths, swapped_prompt_counts, swapped_targets = [], [], [], []
    swaps = []
    for batch_index in range(len(tokens)):
        prompt_count = int(prompt_token_counts[batch_index])
        candidates = eligible_swap_candidates(
            tokens[batch_index],
            prompt_count,
            tag_targets[batch_index],
            tokenizer,
            pool_by_width,
            begin_only,
        )
        if not candidates:
            continue
        candidate = candidates[int(torch.randint(0, len(candidates), (1,)))]
        token_position, gold_tag, original_token_id, width, replacements = candidate
        replacement_token_id = replacements[
            int(torch.randint(0, len(replacements), (1,)))
        ]
        swapped_rows.append(
            apply_same_width_swap(
                tokens, batch_index, token_position, replacement_token_id, tokenizer
            )
        )
        swapped_lengths.append(sequence_lengths[batch_index].clone())
        swapped_prompt_counts.append(prompt_token_counts[batch_index].clone())
        swapped_targets.append(tag_targets[batch_index].clone())
        swaps.append(
            IdentitySwap(
                batch_index,
                len(tokens) + len(swapped_rows) - 1,
                token_position,
                0,
                original_token_id,
                replacement_token_id,
                width,
                gold_tag,
            )
        )
    if begin_only and any(swap.gold_tag != begin_tag("route") for swap in swaps):
        raise RuntimeError("begin-only mode produced a non-begin pair")
    if not swapped_rows:
        return batch, ()
    return (
        (
            torch.cat((tokens, torch.stack(swapped_rows)), dim=0),
            torch.cat((sequence_lengths, torch.stack(swapped_lengths)), dim=0),
            torch.cat((prompt_token_counts, torch.stack(swapped_prompt_counts)), dim=0),
            torch.cat((tag_targets, torch.stack(swapped_targets)), dim=0),
        ),
        tuple(swaps),
    )


def paired_identity_invariance_loss(
    tag_logits: torch.Tensor,
    swaps: Iterable[IdentitySwap],
) -> torch.Tensor:
    """Mean Jensen-Shannon divergence of paired route B/I predictions."""
    swaps = tuple(swaps)
    if not swaps:
        return tag_logits.reshape(-1)[0] * 0.0
    classes = torch.tensor(
        [begin_tag("route"), inside_tag("route")],
        device=tag_logits.device,
        dtype=torch.long,
    )
    losses = []
    for swap in swaps:
        original = tag_logits[
            swap.original_batch_index, swap.token_position, swap.byte_offset
        ].index_select(0, classes)
        paired = tag_logits[
            swap.swapped_batch_index, swap.token_position, swap.byte_offset
        ].index_select(0, classes)
        original_log = F.log_softmax(original.float(), dim=-1)
        paired_log = F.log_softmax(paired.float(), dim=-1)
        original_probability = original_log.exp()
        paired_probability = paired_log.exp()
        midpoint_log = (0.5 * (original_probability + paired_probability)).log()
        losses.append(
            0.5
            * (
                torch.sum(original_probability * (original_log - midpoint_log))
                + torch.sum(paired_probability * (paired_log - midpoint_log))
            )
        )
    return torch.stack(losses).mean().to(dtype=tag_logits.dtype)


def identity_swap_position_counts(swaps: Iterable[IdentitySwap]) -> dict[str, int]:
    """Count generated swaps by gold tag class, for the begin-only hard audit."""
    counts = {"B": 0, "I": 0}
    begin, inside = begin_tag("route"), inside_tag("route")
    for swap in swaps:
        if swap.gold_tag == begin:
            counts["B"] += 1
        elif swap.gold_tag == inside:
            counts["I"] += 1
        else:
            raise ValueError(f"swap has an unexpected gold tag {swap.gold_tag}")
    return counts


def supervised_diagnostic_losses(
    tag_logits: torch.Tensor,
    tag_targets: torch.Tensor,
    original_row_count: int,
) -> tuple[float, float]:
    """Log-only unweighted mean cross-entropy for original vs. swapped rows.

    This never receives a gradient and never participates in the training
    objective; it exists so a human can see whether span damage tracks the
    swapped copy's supervised loss specifically.
    """
    if tag_logits.shape[0] != tag_targets.shape[0]:
        raise ValueError("tag_logits and tag_targets batch sizes differ")
    if not 0 < original_row_count <= tag_logits.shape[0]:
        raise ValueError("original_row_count must be within the batch")

    def _mean_loss(logits_slice: torch.Tensor, targets_slice: torch.Tensor) -> float:
        flat_logits = logits_slice.reshape(-1, logits_slice.shape[-1])
        flat_targets = targets_slice.reshape(-1)
        supervised = flat_targets != -100
        if not bool(supervised.any()):
            return 0.0
        with torch.no_grad():
            return float(
                F.cross_entropy(
                    flat_logits[supervised].float(), flat_targets[supervised]
                )
            )

    base_loss = _mean_loss(
        tag_logits[:original_row_count], tag_targets[:original_row_count]
    )
    if original_row_count == tag_logits.shape[0]:
        return base_loss, 0.0
    swapped_loss = _mean_loss(
        tag_logits[original_row_count:], tag_targets[original_row_count:]
    )
    return base_loss, swapped_loss
