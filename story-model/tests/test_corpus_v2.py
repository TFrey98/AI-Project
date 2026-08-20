import json

import pytest

from story_model.corpus_v2 import (
    CorpusQualityGates,
    build_foundation_corpus,
    discover_foundation_documents,
    split_documents_by_author,
)


def write_cataloged_documents(tmp_path, count=8):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    entries = []

    for index in range(count):
        relative_path = f"author_{index}/work_{index}.txt"
        document_path = raw_dir / relative_path
        document_path.parent.mkdir()
        document_path.write_text(
            (f"Distinct work {index}. Dialogue and narrative.\n" * 20),
            encoding="utf-8",
        )
        entries.append(
            {
                "path": relative_path,
                "title": f"Work {index}",
                "author": f"Author {index}",
                "author_id": f"author_{index}",
                "source": "Test Archive",
                "source_id": str(index),
                "language": "en",
                "rights": "Public domain for test",
                "categories": [
                    "dialogue",
                    "adventure" if index % 2 else "social",
                ],
            }
        )

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "catalog_version": 1,
                "documents": entries,
            }
        ),
        encoding="utf-8",
    )

    return raw_dir, catalog_path


def permissive_gates():
    return CorpusQualityGates(
        minimum_documents=2,
        minimum_authors=2,
        minimum_training_bytes=1,
        minimum_categories=2,
        maximum_author_fraction=0.5,
    )


def test_foundation_split_is_deterministic_and_author_disjoint(tmp_path):
    raw_dir, catalog_path = write_cataloged_documents(tmp_path)
    documents = discover_foundation_documents(raw_dir, catalog_path)
    first_train, first_val = split_documents_by_author(
        documents,
        validation_fraction=0.2,
        seed=1337,
        minimum_validation_authors=2,
    )
    second_train, second_val = split_documents_by_author(
        list(reversed(documents)),
        validation_fraction=0.2,
        seed=1337,
        minimum_validation_authors=2,
    )

    assert [item.relative_path for item in first_train] == [
        item.relative_path for item in second_train
    ]
    assert [item.relative_path for item in first_val] == [
        item.relative_path for item in second_val
    ]
    assert {item.author_id for item in first_train}.isdisjoint(
        {item.author_id for item in first_val}
    )


def test_foundation_build_records_provenance_and_quality_gates(tmp_path):
    raw_dir, catalog_path = write_cataloged_documents(tmp_path)
    output_dir = tmp_path / "processed"
    manifest = build_foundation_corpus(
        input_dir=raw_dir,
        catalog_path=catalog_path,
        output_dir=output_dir,
        validation_fraction=0.2,
        seed=1337,
        minimum_validation_authors=2,
        gates=permissive_gates(),
    )

    assert manifest["version"] == 2
    assert manifest["split_unit"] == "author"
    assert manifest["quality_gates"]["passed"] is True
    assert manifest["summary"]["documents"] == 8
    assert manifest["summary"]["authors"] == 8
    assert manifest["documents"][0]["source"] == "Test Archive"
    assert manifest["documents"][0]["rights"] == "Public domain for test"
    assert (output_dir / "train.txt").is_file()
    assert (output_dir / "val.txt").is_file()
    assert json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    ) == manifest


def test_foundation_discovery_rejects_uncataloged_files(tmp_path):
    raw_dir, catalog_path = write_cataloged_documents(tmp_path)
    (raw_dir / "extra.txt").write_text("Not cataloged.", encoding="utf-8")

    with pytest.raises(ValueError, match="uncataloged"):
        discover_foundation_documents(raw_dir, catalog_path)


def test_foundation_catalog_rejects_duplicate_source_identity(tmp_path):
    raw_dir, catalog_path = write_cataloged_documents(tmp_path, count=2)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["documents"][1]["source_id"] = "0"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate source/source_id"):
        discover_foundation_documents(raw_dir, catalog_path)


def test_foundation_catalog_requires_stable_author_slug(tmp_path):
    raw_dir, catalog_path = write_cataloged_documents(tmp_path, count=2)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["documents"][0]["author_id"] = "Author Zero"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="lowercase stable slug"):
        discover_foundation_documents(raw_dir, catalog_path)


def test_foundation_quality_gate_rejects_small_corpus(tmp_path):
    raw_dir, catalog_path = write_cataloged_documents(tmp_path)

    with pytest.raises(ValueError, match="quality gates failed"):
        build_foundation_corpus(
            input_dir=raw_dir,
            catalog_path=catalog_path,
            output_dir=tmp_path / "processed",
            validation_fraction=0.2,
            seed=1337,
            minimum_validation_authors=2,
        )


def test_foundation_split_requires_enough_distinct_authors(tmp_path):
    raw_dir, catalog_path = write_cataloged_documents(tmp_path, count=3)
    documents = discover_foundation_documents(raw_dir, catalog_path)

    with pytest.raises(ValueError, match="requires more authors"):
        split_documents_by_author(
            documents,
            validation_fraction=0.2,
            seed=1337,
            minimum_validation_authors=3,
        )
