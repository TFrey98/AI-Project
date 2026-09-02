"""Audit Phase 33c encodings and the count-conditioned routing partition."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.expanded_typed_span_resolver import (
    CLARIFY_ACTION,
    EXPANDED_SPLITS,
    GENERATE_ACTION,
    MAX_CANDIDATES,
    RESOLVE_ACTION,
    encode_expanded_record,
    load_expanded_records,
)
from story_model.unified_typed_span_resolver import (
    NO_SUPPORT_OPTION_INDEX,
    encode_unified_record,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--phase31-data-dir",
        type=Path,
        default=Path("data/character/typed_span_resolver"),
    )
    parser.add_argument(
        "--phase32-data-dir",
        type=Path,
        default=Path("data/character/expanded_typed_span_resolver"),
    )
    args = parser.parse_args()

    checkpoint = read_checkpoint(args.checkpoint, map_location="cpu")
    extra = checkpoint.get("extra", {})
    if extra.get("architecture") != "expanded_typed_span_resolver":
        raise ValueError("audit requires the Phase 32 best checkpoint")
    tokenizer = tokenizer_from_dict(extra.get("tokenizer", {}))
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise ValueError("Phase 32 checkpoint does not contain byte-BPE")
    config = extra.get("config")
    if not isinstance(config, dict):
        raise ValueError("Phase 32 checkpoint has no config")
    block_size = int(config["data"]["block_size"])

    actions = Counter()
    eligibility_partition = Counter()
    total = 0
    maximum_shared = 0
    maximum_option = 0
    for dataset_name, data_dir in (
        ("phase31", args.phase31_data_dir),
        ("phase32", args.phase32_data_dir),
    ):
        for split in EXPANDED_SPLITS:
            records = load_expanded_records(data_dir / f"{split}.jsonl")
            for record in records:
                phase32 = encode_expanded_record(record, tokenizer, block_size)
                unified = encode_unified_record(record, tokenizer, block_size)
                if unified.input_ids != phase32.input_ids:
                    raise ValueError(f"shared encoding changed: {record.record_id}")
                if (
                    unified.option_view_input_ids[:MAX_CANDIDATES]
                    != phase32.candidate_view_input_ids
                ):
                    raise ValueError(
                        f"real-candidate views changed: {record.record_id}"
                    )
                if record.expected_action == RESOLVE_ACTION:
                    expected_target = record.selected_candidate_index
                elif record.expected_action == CLARIFY_ACTION:
                    expected_target = NO_SUPPORT_OPTION_INDEX
                elif record.expected_action == GENERATE_ACTION:
                    expected_target = -100
                else:
                    raise ValueError(f"unknown action: {record.expected_action}")
                if unified.option_target != expected_target:
                    raise ValueError(f"wrong option target: {record.record_id}")
                eligible = tuple(
                    index
                    for index, valid in enumerate(
                        unified.option_valid_mask[:MAX_CANDIDATES]
                    )
                    if valid
                )
                if record.expected_action == RESOLVE_ACTION:
                    if not eligible or record.selected_candidate_index not in eligible:
                        raise ValueError(
                            f"selected candidate is not evidence-eligible: "
                            f"{record.record_id}; eligible={eligible}"
                        )
                elif record.expected_action == GENERATE_ACTION and eligible:
                    raise ValueError(
                        f"generate row has evidence-eligible candidates: "
                        f"{record.record_id}; eligible={eligible}"
                    )
                sentinel_valid = unified.option_valid_mask[NO_SUPPORT_OPTION_INDEX]
                expected_sentinel = record.expected_action != GENERATE_ACTION
                if sentinel_valid != expected_sentinel:
                    raise ValueError(f"wrong sentinel mask: {record.record_id}")
                if record.skill == "scene_route" and (
                    record.expected_action == RESOLVE_ACTION
                    or (
                        record.expected_action == CLARIFY_ACTION
                        and record.case == "missing_evidence"
                    )
                ):
                    expected_eligible_count = 2
                elif record.expected_action == RESOLVE_ACTION:
                    expected_eligible_count = 1
                else:
                    expected_eligible_count = 0
                if len(eligible) != expected_eligible_count:
                    raise ValueError(
                        f"unexpected eligibility partition: {record.record_id}; "
                        f"expected_count={expected_eligible_count}, "
                        f"eligible={eligible}"
                    )
                total += 1
                actions[record.expected_action] += 1
                eligibility_partition[
                    (
                        dataset_name,
                        split,
                        record.skill,
                        record.expected_action,
                        len(eligible),
                    )
                ] += 1
                maximum_shared = max(maximum_shared, unified.sequence_tokens)
                maximum_option = max(
                    maximum_option, *unified.option_view_sequence_tokens
                )
            print(f"{dataset_name}/{split}: {len(records):,} rows passed")
    print(f"total rows: {total:,}")
    print(f"actions: {dict(sorted(actions.items()))}")
    print("eligibility partition:")
    for key, count in sorted(eligibility_partition.items()):
        dataset_name, split, skill, action, eligible_count = key
        print(
            f"- {dataset_name}/{split}/{skill}/{action}: "
            f"eligible={eligible_count}, rows={count:,}"
        )
    print(f"maximum shared tokens: {maximum_shared:,}/{block_size:,}")
    print(f"maximum option-view tokens: {maximum_option:,}/{block_size:,}")
    print("routing partition: zero->clarify, one->resolve, two-plus->action head")
    print("Phase 32 shared and real-candidate encodings preserved exactly")


if __name__ == "__main__":
    main()
