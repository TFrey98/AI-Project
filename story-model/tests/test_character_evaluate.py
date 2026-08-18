import math

from story_model.character_data import (
    CHARACTER_CONTROL_TOKENS,
    CharacterContext,
    CharacterProfile,
    ConversationTurn,
    RelationshipState,
    SceneState,
)
from story_model.character_evaluate import evaluate_character_records
from story_model.character_training import CharacterTrainingRecord
from story_model.data import ByteBPETokenizer
from story_model.models.transformer import TransformerLanguageModel


def make_record(
    conversation_id: str,
    context_id: str,
    behavior_tag: str,
) -> CharacterTrainingRecord:
    context = CharacterContext(
        context_id=context_id,
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
            location="Castle",
            time="Night",
            participants=("elara", "traveler"),
        ),
        recent_turns=(
            ConversationTurn(
                role="user",
                speaker_id="traveler",
                text="What did you find?",
            ),
        ),
        target_response="A clue, not yet a conclusion.",
    )
    return CharacterTrainingRecord(
        conversation_id,
        context,
        behavior_tags=(behavior_tag,),
    )


def test_exact_character_evaluation_reports_behavior_categories():
    tokenizer = ByteBPETokenizer.train(
        "A guarded investigator studies a clue in the castle. " * 8,
        vocab_size=272,
    ).with_special_tokens(CHARACTER_CONTROL_TOKENS)
    records = (
        make_record("gate", "gate_001", "voice"),
        make_record("archive", "archive_001", "memory"),
    )
    model = TransformerLanguageModel(
        vocabulary_size=tokenizer.vocab_size,
        block_size=512,
        embedding_dim=16,
        attention_heads=4,
        layers=1,
        feed_forward_dim=32,
        dropout=0.0,
        position_encoding="rope",
    )

    report = evaluate_character_records(
        model,
        records,
        tokenizer,
        block_size=512,
        device="cpu",
    )

    assert report["examples"] == 2
    assert report["tokens"] > 0
    assert math.isfinite(report["loss"])
    assert math.isfinite(report["perplexity"])
    assert set(report["categories"]) == {"memory", "voice"}
    assert report["categories"]["memory"]["examples"] == 1
    assert report["categories"]["voice"]["examples"] == 1
