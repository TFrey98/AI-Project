from pathlib import Path

import torch
import yaml

from story_model.character_data import (
    CHARACTER_CONTROL_TOKENS,
    CharacterContext,
    CharacterProfile,
    ConversationTurn,
    RelationshipState,
    SceneState,
)
from story_model.character_training import (
    CharacterTrainingRecord,
    build_character_dataset,
)
from story_model.checkpoint import read_checkpoint, save_checkpoint
from story_model.data import ByteBPETokenizer
from story_model.models import build_model
from story_model.train import train


def make_record(
    conversation_id: str,
    context_id: str,
    question: str,
    answer: str,
) -> CharacterTrainingRecord:
    context = CharacterContext(
        context_id=context_id,
        character=CharacterProfile(
            character_id="elara",
            name="Elara",
            summary="A guarded investigator.",
            traits=("guarded",),
            voice=("formal",),
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
                text=question,
            ),
        ),
        target_response=answer,
    )
    return CharacterTrainingRecord(conversation_id, context)


def test_character_jsonl_trains_through_main_loop(tmp_path, capsys):
    records = (
        make_record(
            "gate",
            "gate_001",
            "Is there another entrance?",
            "There is, but we move quietly.",
        ),
        make_record(
            "archive",
            "archive_001",
            "Who altered the ledger?",
            "Someone with a key and reason to fear ink.",
        ),
    )
    data_dir = tmp_path / "data"
    build_character_dataset(
        records,
        data_dir,
        validation_fraction=0.5,
        seed=3,
    )

    source_tokenizer = ByteBPETokenizer.train(
        "The guarded investigator studies the castle ledger. " * 8,
        vocab_size=272,
    )
    model_config = {
        "name": "transformer",
        "position_encoding": "rope",
        "normalization": "layernorm",
        "feed_forward_activation": "swiglu",
        "embedding_dim": 16,
        "attention_heads": 4,
        "layers": 1,
        "feed_forward_dim": 32,
        "dropout": 0.0,
    }
    source_model = build_model(
        model_config,
        vocabulary_size=source_tokenizer.vocab_size,
        block_size=64,
    )
    source_path = tmp_path / "source.pt"
    save_checkpoint(
        source_path,
        source_model,
        step=7,
        extra={
            "config": {"model": model_config},
            "tokenizer": source_tokenizer.to_dict(),
        },
    )

    checkpoint_dir = tmp_path / "checkpoints"
    config = {
        "tokenizer": {
            "type": "byte_bpe",
            "vocab_size": source_tokenizer.base_vocab_size,
            "special_tokens": list(CHARACTER_CONTROL_TOKENS),
        },
        "model": model_config,
        "data": {
            "type": "character_jsonl",
            "train_path": str(data_dir / "train.jsonl"),
            "val_path": str(data_dir / "val.jsonl"),
            "manifest_path": str(data_dir / "manifest.json"),
            "block_size": 512,
            "batch_size": 1,
        },
        "train": {
            "seed": 11,
            "learning_rate": 1.0e-3,
            "min_learning_rate": 1.0e-3,
            "warmup_steps": 0,
            "max_steps": 1,
            "gradient_accumulation_steps": 2,
            "gradient_clip": 1.0,
            "weight_decay": 0.0,
            "log_interval": 1,
            "eval_interval": 1,
            "eval_iters": 1,
            "device": "cpu",
        },
        "checkpoint": {
            "dir": str(checkpoint_dir),
            "save_interval": 1,
        },
    }
    config_path = tmp_path / "character.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    train(
        str(config_path),
        warm_start_path=str(source_path),
    )

    output = capsys.readouterr().out
    checkpoint = read_checkpoint(
        checkpoint_dir / "final.pt",
        map_location="cpu",
    )

    assert checkpoint["step"] == 1
    assert "character_dataset_manifest" in checkpoint["extra"]
    assert checkpoint["extra"]["warm_start"][
        "destination_vocabulary_size"
    ] == source_tokenizer.vocab_size + len(
        CHARACTER_CONTROL_TOKENS
    )
    assert "data type: character_jsonl" in output
    assert "loss objective: response_only" in output
    assert "gradient accumulation steps: 2" in output
    assert "checkpoint:" in output
    assert torch.isfinite(
        checkpoint["model_state_dict"]["token_embeddings.weight"]
    ).all()
