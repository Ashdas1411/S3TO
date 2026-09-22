import itertools

from .similarity import cosine_similarity, pairwise_cosine


def _unique_partitions(client_ids, group_size):
    client_ids = sorted(client_ids)
    n = len(client_ids)
    if n % group_size != 0:
        raise ValueError("Number of clients must be divisible by group_size")
    group_count = n // group_size
    seen = set()
    for first_group in itertools.combinations(client_ids, group_size):
        remaining = [c for c in client_ids if c not in first_group]
        for groups in itertools.combinations(remaining, group_size * (group_count - 1)):
            # For the current prototype group_count is 2, so this creates one partition.
            candidate_groups = [tuple(sorted(first_group)), tuple(sorted(groups))]
            canonical = tuple(sorted(candidate_groups))
            if canonical in seen:
                continue
            seen.add(canonical)
            yield list(canonical)


def choose_complementary_groups(histograms, group_size=3):
    """Choose a deterministic partition with low within-group cosine similarity.

    For the two-edge-aggregator prototype, we enumerate all 3+3 partitions and
    select the one with the smallest total within-group average similarity.
    """
    client_ids = sorted(histograms)
    if len(client_ids) % group_size != 0:
        raise ValueError("Client count must be divisible by group_size")
    pairwise = pairwise_cosine(histograms)
    best = None
    for groups in _unique_partitions(client_ids, group_size):
        scores = []
        for group in groups:
            vals = [pairwise[a][b] for a, b in itertools.combinations(group, 2)]
            scores.append(sum(vals) / max(len(vals), 1))
        objective = sum(scores) / len(scores)
        key = (objective, tuple(tuple(g) for g in groups))
        if best is None or key < best[0]:
            best = (key, groups, scores)
    return best[1], best[2], pairwise


def assignment_from_groups(groups):
    assignment = {}
    for edge_index, group in enumerate(groups, start=1):
        for client_id in group:
            assignment[int(client_id)] = f"edge{edge_index}"
    return assignment
