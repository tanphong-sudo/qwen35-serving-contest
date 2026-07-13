from __future__ import annotations

from bisect import bisect_left


def default_capture_sizes(max_capture_size: int) -> tuple[int, ...]:
    if max_capture_size < 1:
        raise ValueError("max_capture_size must be positive")

    sizes = [size for size in (1, 2, 4) if size <= max_capture_size]
    sizes.extend(range(8, min(max_capture_size + 1, 256), 8))
    if max_capture_size >= 256:
        sizes.extend(range(256, max_capture_size + 1, 16))
    if max_capture_size not in sizes:
        sizes.append(max_capture_size)
    return tuple(sorted(set(sizes)))


def cohort_capture_sizes(
    max_capture_size: int,
    *,
    cohort_size: int,
    max_batch_size: int,
) -> tuple[int, ...]:
    if max_capture_size < 1:
        raise ValueError("max_capture_size must be positive")
    if cohort_size < 1:
        raise ValueError("cohort_size must be positive")
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")

    sizes = list(default_capture_sizes(max_capture_size))
    sizes.extend(range(cohort_size, max_batch_size + 1, cohort_size))
    return tuple(sorted(set(sizes)))


def padded_capture_size(
    batch_size: int,
    capture_sizes: tuple[int, ...],
) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not capture_sizes:
        return batch_size

    index = bisect_left(capture_sizes, batch_size)
    if index == len(capture_sizes):
        return batch_size
    return capture_sizes[index]


def project_tbt_ms(
    base_tbt_ms: float,
    batch_size: int,
    base_padded_size: int,
    tuned_padded_size: int,
    graph_sensitive_fraction: float,
) -> float:
    if not 0 <= graph_sensitive_fraction <= 1:
        raise ValueError("graph_sensitive_fraction must be between 0 and 1")
    if min(batch_size, base_padded_size, tuned_padded_size) < 1:
        raise ValueError("batch sizes must be positive")
    if tuned_padded_size > base_padded_size:
        raise ValueError("tuned_padded_size cannot exceed base_padded_size")

    padding_reduction = 1 - tuned_padded_size / base_padded_size
    return base_tbt_ms * (1 - graph_sensitive_fraction * padding_reduction)
