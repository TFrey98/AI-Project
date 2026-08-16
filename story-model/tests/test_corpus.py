import json

import pytest

from story_model.corpus import (
    build_corpus,
    discover_documents,
    normalize_document,
    split_documents,
)


def write_documents(directory) -> None:
    directory.mkdir()
    (directory / "alpha.txt").write_text(
        "Alpha story.\n" * 20,
        encoding="utf-8",
    )
    (directory / "beta.txt").write_text(
        "Beta story.\n" * 18,
        encoding="utf-8",
    )
    (directory / "gamma.txt").write_text(
        "Gamma story.\n" * 16,
        encoding="utf-8",
    )


def test_normalization_strips_gutenberg_wrapper_and_normalizes_unicode():
    raw_text = (
        "Project metadata\r\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\r\n"
        "Café   \r\n"
        "\r\n"
        "A story.\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\r\n"
        "License text\r\n"
    )

    assert normalize_document(raw_text) == "Café\n\nA story.\n"


def test_corpus_build_is_deterministic_and_has_no_document_leakage(
    tmp_path,
):
    input_dir = tmp_path / "raw"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    write_documents(input_dir)

    first_manifest = build_corpus(
        input_dir=input_dir,
        output_dir=first_output,
        validation_fraction=0.25,
        seed=1337,
    )
    second_manifest = build_corpus(
        input_dir=input_dir,
        output_dir=second_output,
        validation_fraction=0.25,
        seed=1337,
    )

    assert first_manifest == second_manifest
    assert (
        first_output / "train.txt"
    ).read_bytes() == (
        second_output / "train.txt"
    ).read_bytes()
    assert (
        first_output / "val.txt"
    ).read_bytes() == (
        second_output / "val.txt"
    ).read_bytes()

    train_paths = {
        entry["path"]
        for entry in first_manifest["documents"]
        if entry["split"] == "train"
    }
    val_paths = {
        entry["path"]
        for entry in first_manifest["documents"]
        if entry["split"] == "val"
    }

    assert train_paths
    assert val_paths
    assert train_paths.isdisjoint(val_paths)
    assert len(train_paths | val_paths) == 3

    saved_manifest = json.loads(
        (first_output / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert saved_manifest == first_manifest


def test_document_discovery_requires_at_least_two_files(tmp_path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    (input_dir / "only.txt").write_text(
        "Only one document.",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="at least two",
    ):
        discover_documents(input_dir)


def test_document_split_rejects_invalid_fraction(tmp_path):
    input_dir = tmp_path / "raw"
    write_documents(input_dir)
    documents = discover_documents(input_dir)

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        split_documents(
            documents,
            validation_fraction=1.0,
            seed=1337,
        )


def test_document_discovery_rejects_duplicate_content(tmp_path):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    duplicate_text = "The same normalized document.\n"
    (input_dir / "first.txt").write_text(
        duplicate_text,
        encoding="utf-8",
    )
    (input_dir / "second.txt").write_text(
        duplicate_text,
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="duplicate normalized corpus documents",
    ):
        discover_documents(input_dir)


def test_document_split_enforces_minimum_validation_documents(
    tmp_path,
):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()

    for index in range(5):
        (input_dir / f"document_{index}.txt").write_text(
            f"Distinct document {index}.\n" * 10,
            encoding="utf-8",
        )

    documents = discover_documents(input_dir)
    training_documents, validation_documents = split_documents(
        documents,
        validation_fraction=0.01,
        seed=1337,
        minimum_validation_documents=3,
    )

    assert len(training_documents) == 2
    assert len(validation_documents) == 3


def test_document_split_rejects_invalid_minimum(tmp_path):
    input_dir = tmp_path / "raw"
    write_documents(input_dir)
    documents = discover_documents(input_dir)

    with pytest.raises(
        ValueError,
        match="minimum_validation_documents must be positive",
    ):
        split_documents(
            documents,
            validation_fraction=0.1,
            seed=1337,
            minimum_validation_documents=0,
        )
