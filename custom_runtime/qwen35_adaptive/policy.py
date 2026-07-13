from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


class DecodeRequest(Protocol):
    request_id: str
    arrival_time: float
    is_prefill_chunk: bool
    max_tokens: int
    next_decode_eligible_step: int

    @property
    def num_output_tokens(self) -> int: ...


def select_completion_cohort(
    requests: Iterable[DecodeRequest],
    decode_window: int,
) -> frozenset[str]:
    if decode_window < 1:
        raise ValueError("decode_window must be positive")

    mature_decoders = [
        request
        for request in requests
        if request.num_output_tokens > 0 and not request.is_prefill_chunk
    ]
    mature_decoders.sort(
        key=lambda request: (
            request.max_tokens - request.num_output_tokens,
            request.arrival_time,
            request.request_id,
        )
    )
    return frozenset(
        request.request_id for request in mature_decoders[:decode_window]
    )


def deferred_decode_step(current_step: int) -> int:
    return current_step + 2


@dataclass(frozen=True)
class CohortDecision:
    selected_request_ids: frozenset[str]
    deferred_request_ids: frozenset[str]
    blocked_until_step: int


def apply_completion_cohort(
    requests: Iterable[DecodeRequest],
    decode_window: int,
    current_step: int,
) -> CohortDecision:
    request_list = list(requests)
    selected_request_ids = select_completion_cohort(request_list, decode_window)
    blocked_until_step = deferred_decode_step(current_step)
    deferred_request_ids: set[str] = set()

    for request in request_list:
        if (
            request.num_output_tokens > 0
            and not request.is_prefill_chunk
            and request.request_id not in selected_request_ids
        ):
            request.next_decode_eligible_step = max(
                request.next_decode_eligible_step,
                blocked_until_step,
            )
            deferred_request_ids.add(request.request_id)

    return CohortDecision(
        selected_request_ids=selected_request_ids,
        deferred_request_ids=frozenset(deferred_request_ids),
        blocked_until_step=blocked_until_step,
    )
