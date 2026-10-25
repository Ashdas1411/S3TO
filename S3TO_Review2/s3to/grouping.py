"""Validated deterministic grouping of FL clients using label histograms."""

import itertools
from collections.abc import Mapping

import numpy as np

from .similarity import pairwise_cosine


def _validate_group_size(group_size):
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, (int, np.integer))
        or group_size <= 0
    ):
        raise ValueError("group_size must be a positive integer")


def _unique_partitions(client_ids, group_size):
    """
    Enumerate every unique, unordered partition into equal-sized groups.

    The smallest remaining client is forced into the next group. This removes
    group-order duplicates and works for any divisible client count.
    """
    _validate_group_size(group_size)

    client_ids = list(client_ids)
    if not client_ids:
        raise ValueError("At least one client is required")
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("Client IDs must be unique")

    client_ids = tuple(sorted(client_ids))

    if len(client_ids) % group_size != 0:
        raise ValueError(
            "Number of clients must be divisible by group_size"
        )

    def generate(remaining):
        if not remaining:
            yield []
            return

        first = remaining[0]

        for others in itertools.combinations(
            remaining[1:], group_size - 1
        ):
            group = (first,) + others
            selected = set(group)
            next_remaining = tuple(
                client for client in remaining
                if client not in selected
            )

            for tail in generate(next_remaining):
                yield [group] + tail

    yield from generate(client_ids)


def _validate_histograms(histograms):
    if not isinstance(histograms, Mapping):
        raise ValueError("histograms must be a mapping of client IDs to vectors")
    if not histograms:
        raise ValueError("At least one client histogram is required")

    clean = {}

    for raw_id, raw_histogram in histograms.items():
        try:
            client_id = int(raw_id)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Invalid client ID: {raw_id!r}"
            ) from exc

        if isinstance(raw_id, (float, np.floating)) and raw_id != client_id:
            raise ValueError(f"Client ID must be an integer: {raw_id!r}")

        if client_id in clean:
            raise ValueError(
                f"Duplicate client ID after normalization: {client_id}"
            )

        try:
            histogram = np.asarray(raw_histogram, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid histogram for client {client_id}"
            ) from exc

        if histogram.ndim != 1 or histogram.size == 0:
            raise ValueError(
                f"Histogram for client {client_id} must be a non-empty vector"
            )

        if not np.all(np.isfinite(histogram)):
            raise ValueError(
                f"Histogram for client {client_id} contains NaN or infinity"
            )

        if np.any(histogram < 0):
            raise ValueError(
                f"Histogram for client {client_id} contains negative counts"
            )

        if histogram.sum() <= 0:
            raise ValueError(
                f"Histogram for client {client_id} has zero total count"
            )

        clean[client_id] = histogram

    lengths = {histogram.size for histogram in clean.values()}
    if len(lengths) != 1:
        raise ValueError("All client histograms must have equal vector lengths")

    return dict(sorted(clean.items()))


def choose_complementary_groups(histograms, group_size=3):
    """
    Minimize average within-group pairwise cosine similarity.

    This preserves the existing S3TO objective. It does not claim that
    low similarity automatically improves model accuracy or network cost.
    """
    _validate_group_size(group_size)
    clean = _validate_histograms(histograms)
    client_ids = list(clean)

    if len(client_ids) % group_size != 0:
        raise ValueError(
            "Number of clients must be divisible by group_size"
        )

    pairwise = pairwise_cosine(clean)
    best = None

    for groups in _unique_partitions(client_ids, group_size):
        scores = []

        for group in groups:
            values = [
                pairwise[a][b]
                for a, b in itertools.combinations(group, 2)
            ]
            scores.append(
                sum(values) / len(values) if values else 0.0
            )

        objective = sum(scores) / len(scores)
        key = (objective, tuple(tuple(group) for group in groups))

        if best is None or key < best[0]:
            best = (key, groups, scores)

    if best is None:
        raise RuntimeError("No valid client partition was generated")

    return best[1], best[2], pairwise


def assignment_from_groups(groups):
    assignment = {}

    for edge_index, group in enumerate(groups, start=1):
        for client_id in group:
            assignment[int(client_id)] = f"edge{edge_index}"

    return assignment
