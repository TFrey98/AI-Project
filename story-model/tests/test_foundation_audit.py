import json

import pytest
import torch

from story_model.checkpoint import save_checkpoint
from story_model.corpus_v2 import CorpusQualityGates, build_foundation_corpus
from story_model.data import ByteBPETokenizer
from story_model.foundation_audit import audit_foundation_corpus
from test_corpus_v2 import write_cataloged_documents


def build_manifest(tmp_path):
    raw_dir, catalog_path = write_cataloged_documents(tmp_path)
    output_dir = tmp_path / "processed"
    build_foundation_corpus(
        input_dir=raw_dir,
        catalog_path=catalog_path,
        output_dir=output_dir,
        validation_fraction=0.2,
        seed=1337,
        minimum_validation_authors=2,
        gates=CorpusQualityGates(
            minimum_documents=2,
            minimum_authors=2,
            minimum_training_bytes=1,
            minimum_categories=2,
            maximum_author_fraction=0.5,
        ),
    )
    return output_dir / "manifest.json"


def write_checkpoint(tmp_path, special_tokens=()):
    path = tmp_path / "best.pt"
    tokenizer = ByteBPETokenizer(
        merges=[],
        special_tokens=special_tokens,
    )
    save_checkpoint(
        path=path,
        model=torch.nn.Linear(2, 2),
        step=10,
        extra={"tokenizer": tokenizer.to_dict()},
    )
    return path


def test_foundation_audit_reports_exact_token_scale(tmp_path):
    manifest_path = build_manifest(tmp_path)
    checkpoint_path = write_checkpoint(tmp_path)
    audit = audit_foundation_corpus(
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        minimum_training_tokens=1,
        minimum_tokens_per_parameter=0.01,
    )

    assert audit["passed"] is True
    assert audit["checkpoint_step"] == 10
    assert audit["vocabulary"] == 256
    assert audit["parameters"] == 6
    assert audit["training_tokens"] == audit["training_utf8_bytes"]


def test_foundation_audit_rejects_character_checkpoint(tmp_path):
    manifest_path = build_manifest(tmp_path)
    checkpoint_path = write_checkpoint(
        tmp_path,
        special_tokens=("<|character|>",),
    )

    with pytest.raises(ValueError, match="pre-character foundation"):
        audit_foundation_corpus(
            manifest_path=manifest_path,
            checkpoint_path=checkpoint_path,
            minimum_training_tokens=1,
            minimum_tokens_per_parameter=0.01,
        )


def test_foundation_audit_rejects_modified_split(tmp_path):
    manifest_path = build_manifest(tmp_path)
    checkpoint_path = write_checkpoint(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = manifest_path.parent / manifest["train"]["path"]
    train_path.write_text("modified", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        audit_foundation_corpus(
            manifest_path=manifest_path,
            checkpoint_path=checkpoint_path,
            minimum_training_tokens=1,
            minimum_tokens_per_parameter=0.01,
        )
