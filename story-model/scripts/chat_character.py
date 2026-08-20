"""Run a local, multi-turn character conversation."""

from __future__ import annotations

import argparse

from story_model.character_chat import (
    CharacterChatSession,
    generate_character_response,
    load_character_runtime,
)
from story_model.character_data import load_character_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--context", required=True)
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
    context = load_character_context(args.context)

    def respond(active_context, response_number):
        return generate_character_response(
            model=runtime.model,
            tokenizer=runtime.tokenizer,
            context=active_context,
            block_size=runtime.block_size,
            device=runtime.device,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + response_number,
            temperature=args.temperature,
            top_k=args.top_k,
            greedy=args.greedy,
        )

    session = CharacterChatSession(context, respond)
    name = context.character.name
    participant = context.relationship.participant_name

    print(f"checkpoint: {args.checkpoint}")
    print(f"completed updates: {runtime.checkpoint_step}")
    print(f"device: {runtime.device}")
    print(f"character: {name}")
    print("commands: /quit")

    if session.has_pending_user_turn:
        pending = session.turns[-1].text
        print(f"{participant}: {pending}")
        generation = session.respond()
        print(f"{name}: {generation.text}")

    while True:
        try:
            user_text = input(f"{participant}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_text == "/quit":
            break
        if not user_text:
            continue

        generation = session.respond(user_text)
        print(f"{name}: {generation.text}")


if __name__ == "__main__":
    main()
