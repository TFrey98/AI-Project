from scripts.phase34g_token_end_premise import token_end_premise_decision


def _row(width, offset, predicted="B", **updates):
    row = {
        "dataset": "phase31_regression",
        "split": "lexical",
        "skill": "scene_route",
        "token_width": width,
        "token_byte_offset": offset,
        "baseline_predicted_class": predicted,
    }
    row.update(updates)
    return row


def _supported_rows():
    rows = []
    rows.extend(_row(2, 1, "I") for _ in range(80))
    rows.extend(_row(2, 1, "B") for _ in range(20))
    rows.extend(_row(3, 1, "I") for _ in range(5))
    rows.extend(_row(3, 1, "B") for _ in range(95))
    return rows


def test_token_end_premise_authorizes_supported_frozen_intervention():
    decision = token_end_premise_decision(_supported_rows())

    assert decision["branch"] == "token_end_geometry_indicated"
    assert decision["training_authorized"] is True
    assert decision["target_error_token_end_capture"] == 80 / 85
    assert decision["width_two_error_token_end_capture"] == 1.0
    assert decision["token_end_error_rate_ratio"] == 16.0


def test_token_end_premise_rejects_unconcentrated_errors():
    rows = []
    rows.extend(_row(2, 1, "I") for _ in range(30))
    rows.extend(_row(2, 1, "B") for _ in range(70))
    rows.extend(_row(3, 1, "I") for _ in range(30))
    rows.extend(_row(3, 1, "B") for _ in range(70))

    decision = token_end_premise_decision(rows)

    assert decision["branch"] == "token_identity_audit_indicated"
    assert decision["training_authorized"] is False
    assert decision["failures"]


def test_token_end_premise_rejects_missing_phase34e_fields():
    rows = _supported_rows()
    del rows[0]["token_byte_offset"]

    decision = token_end_premise_decision(rows)

    assert decision["branch"] == "invalid_token_end_premise"
    assert decision["training_authorized"] is False
    assert "token_byte_offset" in decision["invalid_reasons"][0]


def test_token_end_premise_ignores_non_target_rows():
    rows = _supported_rows()
    rows.extend(
        _row(
            4,
            0,
            "I",
            dataset="phase32",
            split="transfer",
            skill="multi_turn_memory",
        )
        for _ in range(500)
    )

    decision = token_end_premise_decision(rows)

    assert decision["branch"] == "token_end_geometry_indicated"
    assert decision["target"]["gold_begin_count"] == 200


def test_token_end_premise_rejects_input_without_target_rows():
    decision = token_end_premise_decision(
        [_row(2, 1, dataset="phase32", skill="promise_recall")]
    )

    assert decision["branch"] == "invalid_token_end_premise"
    assert decision["training_authorized"] is False
