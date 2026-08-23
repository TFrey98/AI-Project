"""Evaluate a local instruction model or from-scratch checkpoint on
neutral conversation mechanics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from story_model.backbones import GenerationSettings, LocalOpenAIBackbone
from story_model.conversation_gate import (
    load_conversation_gate,
    run_conversation_gate,
)
from story_model.foundation_backbone import FoundationCheckpointBackbone


def main() -> None:
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--model",
        help=(
            "Model name served by the local OpenAI-compatible endpoint. "
            "Mutually exclusive with --checkpoint."
        ),
    )
    source_group.add_argument(
        "--checkpoint",
        help=(
            "Path to one of this project's own checkpoints (e.g. "
            "checkpoints/transformer_foundation_v3/best.pt) to test the "
            "from-scratch model directly, with no server required. "
            "Mutually exclusive with --model."
        ),
    )
    parser.add_argument(
        "--data",
        default="examples/generic_conversation_gate.json",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:11434/v1/chat/completions",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for --checkpoint: auto, cpu, mps, or cuda.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    cases = load_conversation_gate(args.data)

    if args.checkpoint is not None:
        backbone = FoundationCheckpointBackbone(
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
    else:
        backbone = LocalOpenAIBackbone(
            model=args.model,
            endpoint=args.endpoint,
            timeout_seconds=args.timeout,
        )

    results = run_conversation_gate(
        backbone,
        cases,
        GenerationSettings(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        ),
    )

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.case_id}")
        print(result.response.text)

        if result.missing_concepts:
            print(f"missing concepts: {result.missing_concepts}")
        if result.found_forbidden:
            print(f"forbidden phrases: {result.found_forbidden}")

        print(f"manual review: {result.manual_criteria}")
        print()

    passed = sum(result.passed for result in results)
    print(f"automatic conversation gate: {passed}/{len(results)} passed")

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "".join(
                json.dumps(result.to_dict(), sort_keys=True) + "\n"
            ),
            encoding="utf-8",
        )
        print(f"results: {output_path}")

    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
