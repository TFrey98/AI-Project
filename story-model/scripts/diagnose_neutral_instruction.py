"""Diagnose neutral instruction memorization, transfer, and grounding."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from story_model.character_chat import (
    generate_character_response,
    load_character_runtime,
)
from story_model.character_data import serialize_character_prompt
from story_model.character_training import load_character_training_records
from story_model.neutral_diagnostics import (
    context_without_evidence,
    neutral_skill,
    normalized_similarity,
    repetition_metrics,
    stratified_neutral_pairs,
    stratified_neutral_records,
    summarize_diagnostic_rows,
)


def _extra_split(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "extra data must use LABEL=PATH"
        )

    label, path = value.split("=", 1)

    if not label or not label.replace("_", "").isalnum():
        raise argparse.ArgumentTypeError(
            "extra-data label must contain letters, digits, or underscores"
        )
    if not path:
        raise argparse.ArgumentTypeError("extra-data path cannot be empty")

    return label, path


def _generation_data(generation) -> dict:
    metrics = repetition_metrics(generation.text)
    return {
        "response": generation.text,
        "generated_tokens": len(generation.token_ids),
        "prompt_tokens": generation.prompt_tokens,
        "stop_reason": generation.stop_reason,
        **metrics,
    }


def _diagnose_split(
    split: str,
    path: str,
    runtime,
    examples_per_skill: int | None,
    pairs_per_skill: int | None,
    max_new_tokens: int,
    seed: int,
) -> tuple[list[dict], dict]:
    loaded_records = load_character_training_records(path)
    records = (
        stratified_neutral_pairs(loaded_records, pairs_per_skill)
        if pairs_per_skill is not None
        else stratified_neutral_records(
            loaded_records,
            examples_per_skill if examples_per_skill is not None else 3,
        )
    )
    rows = []
    prompt_budget = runtime.block_size - max_new_tokens
    started_at = time.monotonic()

    for index, record in enumerate(records):
        context = replace(record.context, target_response=None)
        ablated_context = context_without_evidence(record.context)
        generation_seed = seed + index
        raw_prompt = serialize_character_prompt(context)
        raw_prompt_tokens = len(runtime.tokenizer.encode(raw_prompt))
        full = generate_character_response(
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            context=context,
            block_size=runtime.block_size,
            device=runtime.device,
            max_new_tokens=max_new_tokens,
            seed=generation_seed,
            greedy=True,
        )
        ablated = generate_character_response(
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            context=ablated_context,
            block_size=runtime.block_size,
            device=runtime.device,
            max_new_tokens=max_new_tokens,
            seed=generation_seed,
            greedy=True,
        )
        reference = record.context.target_response
        assert reference is not None
        reference_similarity = normalized_similarity(full.text, reference)
        ablated_similarity = normalized_similarity(ablated.text, reference)
        repetition = repetition_metrics(full.text)
        context_advantage = reference_similarity - ablated_similarity
        rows.append(
            {
                "split": split,
                "scenario_id": context.context_id,
                "conversation_id": record.conversation_id,
                "skill": neutral_skill(record),
                "behavior_tags": list(record.behavior_tags),
                "seed": generation_seed,
                "reference_response": reference,
                "exact_response": full.text.strip() == reference.strip(),
                "end_stop": full.stop_reason == "end",
                "reference_similarity": reference_similarity,
                "context_changed": full.text != ablated.text,
                "context_advantage": context_advantage,
                "context_helped": context_advantage >= 0.05,
                "prompt_budget": prompt_budget,
                "prompt_tokens": full.prompt_tokens,
                "raw_prompt_tokens": raw_prompt_tokens,
                "prompt_truncated": raw_prompt_tokens > prompt_budget,
                "unique_word_ratio": repetition["unique_word_ratio"],
                "degenerate_loop": repetition["degenerate_loop"],
                "full_context": _generation_data(full),
                "without_evidence": _generation_data(ablated),
            }
        )

        completed = index + 1

        report_interval = (
            pairs_per_skill * 2
            if pairs_per_skill is not None
            else examples_per_skill or 3
        )

        if completed % report_interval == 0 or completed == len(records):
            elapsed = time.monotonic() - started_at
            rate = completed / elapsed if elapsed else 0.0
            remaining = (len(records) - completed) / rate if rate else 0.0
            print(
                f"diagnose {split}: {completed}/{len(records)} "
                f"({completed / len(records):.1%}), "
                f"skill {neutral_skill(record)}, "
                f"elapsed {elapsed:.0f}s, ETA {remaining:.0f}s",
                flush=True,
            )

    return rows, summarize_diagnostic_rows(rows)


def _print_summary(split: str, summary: dict, prompt_budget: int) -> None:
    print(f"{split}: examples {summary['examples']}")
    print(
        f"{split}: exact {summary['exact_response_rate']:.1%}, "
        f"end {summary['end_stop_rate']:.1%}, "
        "reference similarity "
        f"{summary['mean_reference_similarity']:.3f}"
    )
    if "counterfactual_pair_exact_rate" in summary:
        print(
            f"{split}: complete pairs exact "
            f"{summary['counterfactual_pair_exact_rate']:.1%}"
        )
    print(
        f"{split}: loops {summary['degenerate_loop_rate']:.1%}, "
        "unique-word ratio "
        f"{summary['mean_unique_word_ratio']:.3f}"
    )
    print(
        f"{split}: output changed without evidence "
        f"{summary['context_changed_rate']:.1%}, "
        f"context helped {summary['context_helped_rate']:.1%}, "
        "mean advantage "
        f"{summary['mean_context_advantage']:+.3f}"
    )
    prompt = summary["prompt_tokens"]
    raw_prompt = summary["raw_prompt_tokens"]
    print(
        f"{split}: prompt tokens "
        f"{prompt['min']}/{prompt['mean']:.1f}/{prompt['max']} "
        f"(min/mean/max; budget {prompt_budget}); "
        f"raw max {raw_prompt['max']}; "
        f"truncated {summary['truncated_prompt_rate']:.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--train-data",
        default="data/character/neutral_instruction/train.jsonl",
    )
    parser.add_argument(
        "--val-data",
        default="data/character/neutral_instruction/val.jsonl",
    )
    parser.add_argument(
        "--transfer-data",
        default=None,
        help="Optional second held-out split for lexical/paraphrase transfer.",
    )
    parser.add_argument(
        "--extra-data",
        action="append",
        type=_extra_split,
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Additional named diagnostic split; repeat for factorized "
            "transfer axes."
        ),
    )
    parser.add_argument("--device", default="auto")
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument("--examples-per-skill", type=int, default=None)
    sample_group.add_argument(
        "--pairs-per-skill",
        type=int,
        default=None,
        help="Sample complete two-row counterfactual pairs per skill.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runtime = load_character_runtime(args.checkpoint, device=args.device)
    all_rows = []
    summaries = {}

    split_paths = [
        ("train", args.train_data),
        ("val", args.val_data),
    ]

    if args.transfer_data is not None:
        split_paths.append(("transfer", args.transfer_data))

    split_paths.extend(args.extra_data)
    split_labels = [split for split, _ in split_paths]

    if len(split_labels) != len(set(split_labels)):
        parser.error("diagnostic split labels must be unique")

    for split, path in split_paths:
        rows, summary = _diagnose_split(
            split=split,
            path=path,
            runtime=runtime,
            examples_per_skill=args.examples_per_skill,
            pairs_per_skill=args.pairs_per_skill,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        all_rows.extend(rows)
        summaries[split] = summary

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in all_rows
        ),
        encoding="utf-8",
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"checkpoint: {args.checkpoint}")
    print(f"completed updates: {runtime.checkpoint_step}")
    print(f"device: {runtime.device}")
    if args.pairs_per_skill is not None:
        print(f"pairs per skill: {args.pairs_per_skill}")
    else:
        print(f"examples per skill: {args.examples_per_skill or 3}")
    prompt_budget = runtime.block_size - args.max_new_tokens

    for split, _ in split_paths:
        _print_summary(split, summaries[split], prompt_budget)

    print(f"details: {output_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
