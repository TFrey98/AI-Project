import pytest
import torch
from torch import nn

from story_model.character_chat import (
    CharacterChatSession,
    CharacterGeneration,
    generate_character_response,
)
from story_model.character_data import (
    CHARACTER_CONTROL_TOKENS,
    END_MARKER,
    USER_MARKER,
    CharacterContext,
    CharacterProfile,
    ConversationTurn,
    RelationshipState,
    SceneState,
)
from story_model.data import ByteBPETokenizer


def make_context() -> CharacterContext:
    return CharacterContext(
        context_id="chat_001",
        character=CharacterProfile(
            character_id="elara",
            name="Elara",
            summary="A guarded investigator.",
        ),
        relationship=RelationshipState(
            character_id="elara",
            participant_id="traveler",
            participant_name="Traveler",
            attitude="Cautiously cooperative.",
        ),
        scene=SceneState(
            location="Castle gate",
            participants=("elara", "traveler"),
        ),
        recent_turns=(
            ConversationTurn(
                role="user",
                speaker_id="traveler",
                text="What did you find?",
            ),
        ),
    )


class ScheduledModel(nn.Module):
    def __init__(self, vocabulary_size, scheduled_tokens):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.scheduled_tokens = list(scheduled_tokens)
        self.calls = 0

    def forward(self, tokens, targets=None):
        next_id = self.scheduled_tokens[self.calls]
        self.calls += 1
        logits = torch.full(
            (*tokens.shape, self.vocabulary_size),
            -100.0,
            device=tokens.device,
        )
        logits[:, -1, next_id] = 100.0
        return logits, None


def make_tokenizer():
    return ByteBPETokenizer(
        merges=[],
        special_tokens=CHARACTER_CONTROL_TOKENS,
    )


def test_generation_returns_response_only_and_stops_at_end():
    tokenizer = make_tokenizer()
    model = ScheduledModel(
        tokenizer.vocab_size,
        [ord("A"), tokenizer.special_token_ids[END_MARKER]],
    )

    generation = generate_character_response(
        model=model,
        tokenizer=tokenizer,
        context=make_context(),
        block_size=512,
        device="cpu",
        max_new_tokens=16,
        greedy=True,
    )

    assert generation.text == "A"
    assert generation.token_ids == (ord("A"),)
    assert generation.stop_reason == "end"
    assert generation.prompt_tokens <= 496


def test_generation_stops_before_unexpected_control_token():
    tokenizer = make_tokenizer()
    model = ScheduledModel(
        tokenizer.vocab_size,
        [ord("A"), tokenizer.special_token_ids[USER_MARKER]],
    )

    generation = generate_character_response(
        model=model,
        tokenizer=tokenizer,
        context=make_context(),
        block_size=512,
        device="cpu",
        max_new_tokens=16,
        greedy=True,
    )

    assert generation.text == "A"
    assert generation.stop_reason == "control_token"


def test_generation_reserves_room_for_response():
    tokenizer = make_tokenizer()
    model = ScheduledModel(
        tokenizer.vocab_size,
        [tokenizer.special_token_ids[END_MARKER]],
    )

    with pytest.raises(ValueError, match="smaller than block_size"):
        generate_character_response(
            model=model,
            tokenizer=tokenizer,
            context=make_context(),
            block_size=32,
            device="cpu",
            max_new_tokens=32,
            greedy=True,
        )


def test_chat_session_answers_pending_turn_then_accepts_next_user():
    calls = []

    def respond(context, response_number):
        calls.append((context, response_number))
        return CharacterGeneration(
            text=f"answer {response_number}",
            token_ids=(1,),
            prompt_tokens=10,
            stop_reason="end",
            seed=1337 + response_number,
        )

    session = CharacterChatSession(make_context(), respond)

    assert session.has_pending_user_turn
    first = session.respond()
    second = session.respond("And what does it mean?")

    assert first.text == "answer 0"
    assert second.text == "answer 1"
    assert [turn.role for turn in session.turns] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert calls[1][0].recent_turns[-1].text == "And what does it mean?"


def test_chat_session_rejects_two_unanswered_user_turns():
    session = CharacterChatSession(
        make_context(),
        lambda context, number: CharacterGeneration(
            "answer", (1,), 10, "end", number
        ),
    )

    with pytest.raises(ValueError, match="pending turn"):
        session.respond("Another question")
