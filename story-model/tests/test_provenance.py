from story_model.provenance import (
    canonical_json_sha256,
    training_fingerprints,
)


def test_canonical_json_fingerprint_ignores_mapping_order():
    first = {"b": [2, 3], "a": 1}
    second = {"a": 1, "b": [2, 3]}

    assert canonical_json_sha256(first) == canonical_json_sha256(second)


def test_training_fingerprints_include_exact_split_hashes():
    tokenizer = {"type": "byte_bpe", "merges": [[1, 2]]}
    manifest = {
        "version": 2,
        "train": {"sha256": "train-hash"},
        "val": {"sha256": "val-hash"},
    }

    fingerprints = training_fingerprints(tokenizer, manifest)

    assert len(fingerprints["tokenizer_sha256"]) == 64
    assert len(fingerprints["corpus_manifest_sha256"]) == 64
    assert fingerprints["train_sha256"] == "train-hash"
    assert fingerprints["val_sha256"] == "val-hash"
