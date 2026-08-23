from story_model.backbones import GenerationSettings, ScriptedBackbone
from story_model.character_chat import CharacterChatSession
from story_model.character_data import (
    CharacterContext,
    CharacterProfile,
    ConversationTurn,
    MemoryRecord,
    RelationshipState,
    SceneState,
    WorldFact,
)
from story_model.conversation_runtime import (
    InstructCharacterResponder,
    build_character_chat_messages,
    build_character_system_prompt,
)


def make_context():
    return CharacterContext(
        context_id="neutral_chat",
        character=CharacterProfile(
            character_id="guide",
            name="Guide",
            summary="A practical expedition guide.",
        ),
        relationship=RelationshipState(
            character_id="guide",
            participant_id="traveler",
            participant_name="Traveler",
            attitude="Cooperative.",
        ),
        scene=SceneState(
            location="Castle gate",
            situation="The gate is barred.",
            participants=("guide", "traveler"),
        ),
        memories=(
            MemoryRecord(
                memory_id="visible",
                owner_id="guide",
                content="The eastern tunnel was open yesterday.",
            ),
            MemoryRecord(
                memory_id="private_other",
                owner_id="rival",
                content="The rival hid a key beneath the well.",
            ),
        ),
        world_facts=(
            WorldFact(
                fact_id="known",
                content="The main gate is barred.",
                known_by=("guide",),
            ),
            WorldFact(
                fact_id="unknown",
                content="A private door exists in the tower.",
                known_by=("rival",),
            ),
        ),
        recent_turns=(
            ConversationTurn(
                role="user",
                speaker_id="traveler",
                text="Can we get inside?",
            ),
        ),
    )


def test_instruct_prompt_includes_only_visible_knowledge():
    prompt = build_character_system_prompt(make_context())

    assert "eastern tunnel" in prompt
    assert "main gate is barred" in prompt
    assert "private internal context" in prompt
    assert "key beneath the well" not in prompt
    assert "private door exists" not in prompt


def test_character_messages_preserve_chat_roles():
    messages = build_character_chat_messages(make_context())

    assert [message.role for message in messages] == ["system", "user"]
    assert messages[-1].content == "Can we get inside?"


def test_instruct_responder_increments_seed_per_response():
    backend = ScriptedBackbone(
        ["Use the eastern tunnel.", "It was open yesterday."]
    )
    responder = InstructCharacterResponder(
        backend,
        GenerationSettings(seed=100),
    )
    session = CharacterChatSession(make_context(), responder)

    first = session.respond()
    second = session.respond("When was it last open?")

    assert first.text == "Use the eastern tunnel."
    assert second.text == "It was open yesterday."
    assert [call[1].seed for call in backend.calls] == [100, 101]
    assert [message.role for message in backend.calls[1][0]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
