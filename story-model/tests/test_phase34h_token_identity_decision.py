from scripts.phase34h_token_identity_decision import token_identity_decision


def _row(split, token_id, first_byte, predicted="B", **updates):
    row = {
        "dataset": "phase31_regression",
        "split": split,
        "skill": "scene_route",
        "baseline_predicted_class": predicted,
        "token_id": token_id,
        "token_bytes_hex": bytes((first_byte, 120)).hex(),
        "token_utf8": bytes((first_byte, 120)).decode("ascii"),
        "token_width": 2,
        "token_byte_offset": 0,
        "first_byte": first_byte,
    }
    row.update(updates)
    return row


def _group(split, token_id, first_byte, starts, errors):
    return [
        _row(
            split,
            token_id,
            first_byte,
            "I" if index < errors else "B",
        )
        for index in range(starts)
    ]


def _statistics(token_ids, conflict_id=None):
    result = {}
    for token_id in token_ids:
        typed = (
            {"B:route": 1, "I:route": 20}
            if token_id == conflict_id
            else {"B:route": 20, "I:route": 1}
        )
        result[str(token_id)] = {
            "reference": {
                "train": {
                    "prompt_token_occurrences": 80,
                    "labels": {"O": 40, "B": 16, "I": 24},
                    "typed_labels": typed,
                },
                "val": {
                    "prompt_token_occurrences": 20,
                    "labels": {"O": 10, "B": 4, "I": 6},
                    "typed_labels": {},
                },
                "combined": {
                    "prompt_token_occurrences": 100,
                    "labels": {"O": 50, "B": 20, "I": 30},
                    "typed_labels": typed,
                }
            },
            "familiar_phase31_train_val": {
                "gold_begin_count": 20,
                "begin_as_inside_count": 0,
                "begin_as_inside_rate": 0.0,
                "exact_begin_rate": 1.0,
            },
        }
    return result


def test_train_label_prior_conflict_takes_precedence():
    rows = []
    for split in ("lexical", "transfer"):
        rows.extend(_group(split, 300, 113, 60, 50))
        rows.extend(_group(split, 301, 97, 40, 5))

    decision = token_identity_decision(
        rows, _statistics((300, 301), conflict_id=300)
    )

    assert decision["branch"] == "train_label_prior_conflict_indicated"
    assert decision["reference_inside_prior"]["passed"] is True
    assert decision["reference_inside_prior"]["token_ids"] == [300]
    assert decision["training_authorized"] is False


def test_exact_token_identity_must_confirm_from_lexical_to_transfer():
    rows = []
    for split in ("lexical", "transfer"):
        rows.extend(_group(split, 300, 113, 60, 40))
        rows.extend(_group(split, 301, 97, 40, 5))

    decision = token_identity_decision(rows, _statistics((300, 301)))

    assert decision["branch"] == "token_identity_concentration_indicated"
    confirmation = decision["lexical_to_transfer_token_confirmation"]
    assert confirmation["passed"] is True
    assert confirmation["discovered_values"] == [300]


def test_first_byte_can_confirm_when_exact_token_does_not_transfer():
    rows = []
    rows.extend(_group("lexical", 300, 113, 60, 40))
    rows.extend(_group("lexical", 400, 97, 40, 5))
    rows.extend(_group("transfer", 301, 113, 60, 40))
    rows.extend(_group("transfer", 401, 97, 40, 5))

    decision = token_identity_decision(
        rows, _statistics((300, 301, 400, 401))
    )

    assert decision["branch"] == "byte_identity_concentration_indicated"
    assert (
        decision["lexical_to_transfer_token_confirmation"]["passed"]
        is False
    )
    assert (
        decision["lexical_to_transfer_first_byte_confirmation"]["passed"]
        is True
    )


def test_diffuse_errors_indicate_factorized_boundary_type_head():
    rows = []
    for split in ("lexical", "transfer"):
        rows.extend(_group(split, 300, 113, 50, 10))
        rows.extend(_group(split, 301, 97, 50, 10))

    decision = token_identity_decision(rows, _statistics((300, 301)))

    assert decision["branch"] == "factorized_boundary_type_head_indicated"
    assert decision["training_authorized"] is False
    assert decision["checkpoint_promotion_authorized"] is False


def test_audit_rejects_missing_token_statistics():
    rows = []
    for split in ("lexical", "transfer"):
        rows.extend(_group(split, 300, 113, 60, 20))

    decision = token_identity_decision(rows, {})

    assert decision["branch"] == "invalid_token_identity_audit"
    assert decision["invalid_reasons"]


def test_audit_requires_supported_width_two_offset_zero_target():
    rows = _group("lexical", 300, 113, 50, 20)

    decision = token_identity_decision(rows, _statistics((300,)))

    assert decision["branch"] == "invalid_token_identity_audit"
    assert "target starts" in decision["invalid_reasons"][-1]
