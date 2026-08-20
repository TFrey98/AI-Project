"""Generate repeatable candidates for held-out human review scenarios."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from story_model.character_chat import (
    generate_character_response,
    load_character_runtime,
)
from story_model.character_training import load_character_training_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    runtime = load_character_runtime(
        args.checkpoint,
        device=args.device,
    )
    records = load_character_training_records(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    for index, record in enumerate(records):
        context = record.context
        latest_user_turn = next(
            (
                turn.text
                for turn in reversed(context.recent_turns)
                if turn.role == "user"
            ),
            None,
        )
        generation = generate_character_response(
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            context=replace(context, target_response=None),
            block_size=runtime.block_size,
            device=runtime.device,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + index,
            temperature=args.temperature,
            top_k=args.top_k,
            greedy=args.greedy,
        )
        lines.append(
            json.dumps(
                {
                    "scenario_id": context.context_id,
                    "conversation_id": record.conversation_id,
                    "behavior_tags": list(record.behavior_tags),
                    "seed": generation.seed,
                    "prompt_tokens": generation.prompt_tokens,
                    "generated_tokens": len(generation.token_ids),
                    "stop_reason": generation.stop_reason,
                    "user_turn": latest_user_turn,
                    "reference_response": context.target_response,
                    "generated_response": generation.text,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"checkpoint: {args.checkpoint}")
    print(f"data: {args.data}")
    print(f"scenarios: {len(records):,}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
