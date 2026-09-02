from types import SimpleNamespace

from scripts.train_unified_typed_span_resolver import (
    _checkpoint_selection,
    _checkpoint_slot,
    _stratified_indices,
)


def _record(skill, action, case, width):
    return SimpleNamespace(
        skill=skill,
        expected_action=action,
        case=case,
        candidates=tuple(range(width)),
    )


def test_stratified_panel_is_fixed_and_covers_every_cell():
    records = tuple(
        [_record("route", "resolve", "supported", 2) for _ in range(10)]
        + [_record("route", "clarify", "missing", 2) for _ in range(3)]
        + [_record("fact", "resolve", "supported", 4) for _ in range(8)]
    )

    first = _stratified_indices(records, examples_per_cell=4)
    second = _stratified_indices(records, examples_per_cell=4)

    assert first == second
    assert first == (0, 3, 6, 9, 10, 11, 12, 13, 15, 18, 20)
    assert {
        (
            records[index].skill,
            records[index].expected_action,
            records[index].case,
            len(records[index].candidates),
        )
        for index in first
    } == {
        ("route", "resolve", "supported", 2),
        ("route", "clarify", "missing", 2),
        ("fact", "resolve", "supported", 4),
    }


def _phase31_metrics(
    *,
    structured=1.0,
    resolve_option=1.0,
    raw_candidate=1.0,
    sentinel=1.0,
    multi_action=1.0,
    mode=1.0,
):
    return {
        "structured_accuracy": structured,
        "resolve_option_accuracy": resolve_option,
        "raw_candidate_top1_accuracy": raw_candidate,
        "clarify_sentinel_accuracy": sentinel,
        "multi_candidate_action_accuracy": multi_action,
        "mode_accuracy": mode,
    }


def _selection(metrics, loss):
    return _checkpoint_selection(
        metrics,
        loss,
        structured_floor=0.95,
        resolve_option_floor=0.995,
        sentinel_floor=0.95,
        multi_action_floor=0.95,
        mode_floor=0.98,
    )


def test_checkpoint_selection_never_prefers_ineligible_lower_loss():
    eligible, eligible_key = _selection(_phase31_metrics(), loss=0.25)
    regressed, regressed_key = _selection(
        _phase31_metrics(structured=0.94),
        loss=0.01,
    )

    assert eligible
    assert not regressed
    assert eligible_key > regressed_key


def test_checkpoint_selection_rejects_hidden_raw_ranking_regression():
    eligible, _ = _selection(_phase31_metrics(), loss=0.25)
    regressed, _ = _selection(
        _phase31_metrics(raw_candidate=0.99),
        loss=0.01,
    )

    assert eligible
    assert not regressed


def test_checkpoint_selection_rejects_ambiguity_action_regression():
    eligible, _ = _selection(_phase31_metrics(), loss=0.25)
    regressed, _ = _selection(
        _phase31_metrics(multi_action=0.94),
        loss=0.01,
    )

    assert eligible
    assert not regressed


def test_checkpoint_selection_uses_loss_among_eligible_updates():
    first_eligible, first_key = _selection(_phase31_metrics(), loss=0.25)
    second_eligible, second_key = _selection(_phase31_metrics(), loss=0.20)

    assert first_eligible and second_eligible
    assert second_key > first_key


def test_ineligible_checkpoint_uses_diagnostic_slot_only():
    eligible, eligible_key = _selection(_phase31_metrics(), loss=0.25)
    ineligible, ineligible_key = _selection(
        _phase31_metrics(sentinel=0.75),
        loss=0.01,
    )

    assert _checkpoint_slot(
        eligible, eligible_key, best_key=None, diagnostic_key=None
    ) == "best"
    assert _checkpoint_slot(
        ineligible, ineligible_key, best_key=None, diagnostic_key=None
    ) == "diagnostic"
    assert _checkpoint_slot(
        ineligible,
        ineligible_key,
        best_key=eligible_key,
        diagnostic_key=None,
    ) == "diagnostic"
