"""Audit Phase 34 offset annotations without training a proposer."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from story_model.checkpoint import read_checkpoint
from story_model.data import ByteBPETokenizer, tokenizer_from_dict
from story_model.expanded_typed_span_resolver import (
    EXPANDED_SPLITS,
    RESOLVE_ACTION,
    load_expanded_records,
)
from story_model.explicit_offset_candidate_proposer import (
    encode_proposal_record,
    gold_evidence_spans,
    proposal_record_is_eligible,
    proposal_source_width,
    proposed_candidates,
)
from story_model.unified_typed_span_resolver import (
    SUPPORT_MASK_VERSION,
    candidate_is_evidence_eligible,
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
    if extra.get("architecture") != "unified_typed_span_resolver":
        raise ValueError("audit requires a Phase 33c checkpoint")
    if extra.get("support_mask_version") != SUPPORT_MASK_VERSION:
        raise ValueError("checkpoint predates Phase 33c")
    if extra.get("checkpoint_eligible") is not True:
        raise ValueError("audit requires an eligible Phase 33c best checkpoint")
    tokenizer = tokenizer_from_dict(extra.get("tokenizer", {}))
    if not isinstance(tokenizer, ByteBPETokenizer):
        raise ValueError("Phase 34 audit requires byte-BPE")
    config = extra.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no config")
    block_size = int(config["data"]["block_size"])

    partition = Counter()
    total = 0
    excluded = 0
    occurrence_total = 0
    unique_candidate_total = 0
    repeated_occurrence_rows = 0
    maximum_tokens = 0
    for dataset_name, data_dir in (
        ("phase31", args.phase31_data_dir),
        ("phase32", args.phase32_data_dir),
    ):
        for split in EXPANDED_SPLITS:
            records = load_expanded_records(data_dir / f"{split}.jsonl")
            passed = 0
            for record in records:
                if not proposal_record_is_eligible(record):
                    excluded += 1
                    partition[
                        (
                            dataset_name,
                            split,
                            record.skill,
                            record.case,
                            "excluded",
                        )
                    ] += 1
                    continue
                spans = gold_evidence_spans(record)
                encoded = encode_proposal_record(
                    record, tokenizer, block_size
                )
                for span in spans:
                    span.validate_prompt(record.prompt)
                expected_texts = {
                    candidate.text
                    for index, candidate in enumerate(record.candidates)
                    if candidate_is_evidence_eligible(record, index)
                }
                actual_candidates = proposed_candidates(record, spans)
                actual_texts = {candidate.text for candidate in actual_candidates}
                if actual_texts != expected_texts:
                    raise ValueError(
                        f"candidate roundtrip changed {record.record_id}: "
                        f"expected={sorted(expected_texts)}, "
                        f"actual={sorted(actual_texts)}"
                    )
                if (
                    record.expected_action == RESOLVE_ACTION
                    and record.expected_value not in actual_texts
                ):
                    raise ValueError(
                        f"resolve answer is absent from offsets: {record.record_id}"
                    )
                if len(encoded.prompt_byte_tags) != len(
                    record.prompt.encode("utf-8")
                ):
                    raise ValueError(
                        f"byte labels changed prompt length: {record.record_id}"
                    )
                occurrence_total += len(spans)
                unique_candidate_total += len(actual_candidates)
                repeated_occurrence_rows += int(len(spans) > len(actual_candidates))
                maximum_tokens = max(maximum_tokens, encoded.sequence_tokens)
                partition[
                    (
                        dataset_name,
                        split,
                        record.skill,
                        record.case,
                        len(actual_candidates),
                    )
                ] += 1
                total += 1
                passed += 1
            print(f"{dataset_name}/{split}: {passed:,} proposer rows passed")
    print(f"proposal rows: {total:,}")
    print(f"wrong_type rows retained only for oracle gate: {excluded:,}")
    print(f"explicit occurrences: {occurrence_total:,}")
    print(f"unique candidate values: {unique_candidate_total:,}")
    print(f"rows with repeated candidate occurrences: {repeated_occurrence_rows:,}")
    print(f"proposal byte width: {proposal_source_width(tokenizer)}")
    print(f"maximum proposal input: {maximum_tokens:,}/{block_size:,} tokens")
    print("candidate-count partition:")
    for key, count in sorted(partition.items()):
        dataset, split, skill, case, candidate_count = key
        print(
            f"- {dataset}/{split}/{skill}/{case}: "
            f"candidates={candidate_count}, rows={count:,}"
        )
    print("explicit-offset candidate proposal audit: passed")


if __name__ == "__main__":
    main()
