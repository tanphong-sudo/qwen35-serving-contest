from dataclasses import dataclass

import pytest

from custom_runtime.qwen35_adaptive.policy import (
    apply_completion_cohort,
    deferred_decode_step,
    select_completion_cohort,
)


@dataclass
class FakeRequest:
    request_id: str
    arrival_time: float
    num_output_tokens: int
    is_prefill_chunk: bool = False
    max_tokens: int = 200
    next_decode_eligible_step: int = 0


def test_selects_most_complete_decode_requests_first():
    requests = [
        FakeRequest("new", 3.0, 1),
        FakeRequest("old-a", 1.0, 150),
        FakeRequest("old-b", 2.0, 150),
        FakeRequest("middle", 0.0, 80),
    ]

    assert select_completion_cohort(requests, 2) == frozenset({"old-a", "old-b"})


def test_selects_fewest_remaining_tokens_across_different_output_limits():
    requests = [
        FakeRequest("short", 2.0, 20, max_tokens=30),
        FakeRequest("long", 1.0, 150, max_tokens=200),
    ]

    assert select_completion_cohort(requests, 1) == frozenset({"short"})


def test_prefill_and_first_token_requests_do_not_consume_decode_window():
    requests = [
        FakeRequest("prefill", 0.0, 0, is_prefill_chunk=True),
        FakeRequest("first-token", 1.0, 0),
        FakeRequest("decoder", 2.0, 1),
    ]

    assert select_completion_cohort(requests, 1) == frozenset({"decoder"})


def test_uses_arrival_and_request_id_as_stable_tiebreakers():
    requests = [
        FakeRequest("b", 1.0, 10),
        FakeRequest("a", 1.0, 10),
        FakeRequest("older", 0.0, 10),
    ]

    assert select_completion_cohort(requests, 2) == frozenset({"older", "a"})


def test_rejects_non_positive_decode_window():
    with pytest.raises(ValueError, match="decode_window must be positive"):
        select_completion_cohort([], 0)


def test_defers_for_exactly_one_scheduler_turn():
    assert deferred_decode_step(10) == 12


def test_applies_deferral_only_to_mature_non_cohort_decoders():
    selected = FakeRequest("selected", 0.0, 190)
    deferred = FakeRequest("deferred", 1.0, 10)
    first_token = FakeRequest("first-token", 2.0, 0)
    prefill = FakeRequest("prefill", 3.0, 0, is_prefill_chunk=True)

    decision = apply_completion_cohort(
        [selected, deferred, first_token, prefill],
        decode_window=1,
        current_step=7,
    )

    assert decision.selected_request_ids == frozenset({"selected"})
    assert decision.deferred_request_ids == frozenset({"deferred"})
    assert decision.blocked_until_step == 9
    assert selected.next_decode_eligible_step == 0
    assert deferred.next_decode_eligible_step == 9
    assert first_token.next_decode_eligible_step == 0
    assert prefill.next_decode_eligible_step == 0


def test_preserves_stricter_existing_decode_eligibility():
    request = FakeRequest(
        "already-blocked",
        0.0,
        10,
        next_decode_eligible_step=20,
    )

    apply_completion_cohort(
        [FakeRequest("selected", 0.0, 100), request],
        decode_window=1,
        current_step=7,
    )

    assert request.next_decode_eligible_step == 20
