"""Chat through a coherent instruction model running on this computer."""

from __future__ import annotations

import argparse

from story_model.backbones import GenerationSettings, LocalOpenAIBackbone
from story_model.character_chat import CharacterChatSession
from story_model.character_data import load_character_context
from story_model.conversation_runtime import InstructCharacterResponder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:11434/v1/chat/completions",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    context = load_character_context(args.context)
    backbone = LocalOpenAIBackbone(
        model=args.model,
        endpoint=args.endpoint,
        timeout_seconds=args.timeout,
    )
    responder = InstructCharacterResponder(
        backbone,
        GenerationSettings(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        ),
    )
    session = CharacterChatSession(context, responder)
    name = context.character.name
    participant = context.relationship.participant_name

    print(f"backend: local OpenAI-compatible server")
    print(f"endpoint: {args.endpoint}")
    print(f"model: {args.model}")
    print(f"character: {name}")
    print("commands: /quit")

    if session.has_pending_user_turn:
        pending = session.turns[-1].text
        print(f"{participant}: {pending}")
        response = session.respond()
        print(f"{name}: {response.text}")

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

        response = session.respond(user_text)
        print(f"{name}: {response.text}")


if __name__ == "__main__":
    main()
