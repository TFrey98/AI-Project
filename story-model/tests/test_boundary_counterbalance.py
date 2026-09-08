import hashlib
import json
from copy import deepcopy

import pytest

from story_model.boundary_counterbalance import (
    build_boundary_counterbalance_dataset,
    build_counterbalance_records,
    candidate_focus_tokens,
    candidate_value_token_ids,
    focus_boundary_counts,
    scene_route_pairs,
    select_focus_token_pools,
    validate_counterbalance_manifest,
    validate_phase34k_premise,
)
from story_model.data import ByteBPETokenizer
from story_model.expanded_typed_span_resolver import (
    EXPANDED_CONTROL_TOKENS,
    ExpandedCandidate,
    ExpandedResolverRecord,
    save_expanded_records,
)
from story_model.provenance import canonical_json_sha256


def _tokenizer():
    merges = [
        (ord(left), ord(right))
        for left, right in ("ab", "cd", "ef", "gh", "ij", "kl")
    ]
    return ByteBPETokenizer(merges=merges).with_special_tokens(
        EXPANDED_CONTROL_TOKENS
    )


def _pair(split):
    candidates = (
        ExpandedCandidate("eastern tunnel", "route"),
        ExpandedCandidate("south stair", "route"),
    )
    prompts = (
        "Routes: eastern tunnel and south stair. Eastern tunnel is blocked.",
        "Routes: eastern tunnel and south stair. South stair is blocked.",
    )
    rows = []
    for side, selected in enumerate((1, 0)):
        rows.append(
            ExpandedResolverRecord(
                record_id=f"source-{split}-{side}",
                source_context_id=f"context-{split}-{side}",
                conversation_id=f"conversation-{split}",
                split=split,
                skill="scene_route",
                case="supported",
                prompt=prompts[side],
                expected_action="resolve",
                expected_type="route",
                candidates=candidates,
                selected_candidate_index=selected,
                response_template="Take the <|resolved_value|>.",
                expected_value=candidates[selected].text,
                alternative_value=candidates[1 - selected].text,
                source_phase="phase31",
            )
        )
    return tuple(rows)


def _phase34k_summary():
    return {
        "context_path_intervention_version": 1,
        "focus_token_ids": {"or": 270, "ro": 357},
        "models": {
            "phase34d": {
                "checkpoint": {"tokenizer_sha256": "tokenizer-sha"}
            },
            "phase34i": {
                "checkpoint": {"tokenizer_sha256": "tokenizer-sha"}
            },
        },
        "decision": {
            "branch": "local_token_state_dominant",
            "training_authorized": False,
            "full_phase34_evaluation_authorized": False,
            "checkpoint_promotion_authorized": False,
        },
    }


def test_phase34k_premise_requires_local_state_and_frozen_authorizations():
    premise = validate_phase34k_premise(_phase34k_summary())

    assert premise["focus_token_ids"] == {"or": 270, "ro": 357}
    assert premise["tokenizer_sha256"] == "tokenizer-sha"

    changed = _phase34k_summary()
    changed["decision"]["training_authorized"] = True
    with pytest.raises(ValueError, match="training_authorized"):
        validate_phase34k_premise(changed)


def test_counterbalance_uses_disjoint_focus_pools_and_balances_each_identity():
    tokenizer = _tokenizer()
    pairs = {
        split: scene_route_pairs(_pair(split))
        for split in ("train", "val", "lexical", "paraphrase", "transfer")
    }
    pools = select_focus_token_pools(
        tokenizer,
        pairs,
        excluded_token_ids=(),
        train_count=1,
        heldout_count=1,
        block_size=256,
        seed=7,
    )
    records, reports = build_counterbalance_records(
        pairs, pools, tokenizer, 256
    )

    assert {token_id for token_id, _ in pools["train"]}.isdisjoint(
        {token_id for token_id, _ in pools["heldout"]}
    )
    for split, rows in records.items():
        assert len(rows) == 2
        token_id = pools[reports[split]["focus_pool"]][0][0]
        counts = focus_boundary_counts(rows, tokenizer, 256, token_id)
        assert counts["B"] == counts["I"]
        assert counts["B"] > 0


def test_candidate_focus_tokens_are_printable_atomic_two_byte_tokens():
    tokenizer = _tokenizer()
    candidates = candidate_focus_tokens(tokenizer)

    assert len(candidates) == 6
    assert {text for _, text in candidates} == {
        "ab",
        "cd",
        "ef",
        "gh",
        "ij",
        "kl",
    }


def test_focus_tokens_exclude_generated_route_filler_collisions():
    tokenizer = ByteBPETokenizer(
        merges=[
            (ord("o"), ord("u")),
            (ord("t"), ord("h")),
            (ord("a"), ord("b")),
        ]
    ).with_special_tokens(EXPANDED_CONTROL_TOKENS)

    texts = {text for _, text in candidate_focus_tokens(tokenizer)}

    assert "ou" not in texts
    assert "th" not in texts
    assert "ab" in texts


def test_training_value_token_inventory_is_explicit():
    tokenizer = _tokenizer()
    ids = candidate_value_token_ids(_pair("train"), tokenizer)

    assert ids
    assert all(type(token_id) is int for token_id in ids)


def test_scene_route_pair_contract_rejects_an_incomplete_conversation():
    with pytest.raises(ValueError, match="is not a pair"):
        scene_route_pairs(_pair("train")[:1])


def test_complete_dataset_freezes_hashes_balance_and_value_disjointness(tmp_path):
    tokenizer = _tokenizer()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_files = {}
    source_records = {}
    for split in ("train", "val", "lexical", "paraphrase", "transfer"):
        path = source_dir / f"{split}.jsonl"
        rows = _pair(split)
        save_expanded_records(rows, path)
        source_files[split] = path
        source_records[split] = rows

    summary = _phase34k_summary()
    tokenizer_hash = canonical_json_sha256(tokenizer.to_dict())
    for model in summary["models"].values():
        model["checkpoint"]["tokenizer_sha256"] = tokenizer_hash
    for split in ("lexical", "transfer"):
        path = source_files[split]
        summary.setdefault("data_files", {})[split] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(source_records[split]),
        }
    summary_path = tmp_path / "phase34k.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output_dir = tmp_path / "counterbalance"

    manifest = build_boundary_counterbalance_dataset(
        source_records,
        tokenizer,
        summary,
        output_dir,
        source_files,
        summary_path,
        train_focus_tokens=1,
        heldout_focus_tokens=1,
        block_size=256,
        seed=11,
    )
    pools = validate_counterbalance_manifest(manifest, output_dir, tokenizer)

    assert set(pools) == {"train", "heldout"}
    assert manifest["value_token_disjointness"][
        "heldout_focus_ids_in_training_values"
    ] == []
    assert manifest["focus_collision_policy"][
        "global_positive_counts_must_equal_assigned_counts"
    ] is True

    changed = deepcopy(manifest)
    changed["value_token_disjointness"]["training_value_token_count"] += 1
    with pytest.raises(ValueError, match="value-token count changed"):
        validate_counterbalance_manifest(changed, output_dir, tokenizer)
