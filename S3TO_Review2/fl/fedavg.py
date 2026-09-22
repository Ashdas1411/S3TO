from collections import OrderedDict

import torch


def fedavg(state_dicts, sample_counts):
    if not state_dicts:
        raise ValueError("fedavg requires at least one state_dict")
    if len(state_dicts) != len(sample_counts):
        raise ValueError("state_dicts and sample_counts must have the same length")

    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("Total sample count must be positive")

    result = OrderedDict()
    keys = state_dicts[0].keys()
    for key in keys:
        first = state_dicts[0][key]
        if torch.is_floating_point(first):
            result[key] = sum(
                state_dict[key].float() * (count / total)
                for state_dict, count in zip(state_dicts, sample_counts)
            ).to(first.dtype)
        else:
            result[key] = first.clone()
    return result
