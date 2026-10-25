from collections import OrderedDict
from numbers import Integral

import torch


def fedavg(state_dicts, sample_counts):
    """Sample-weighted FedAvg with strict structure/count validation."""
    if not state_dicts:
        raise ValueError("fedavg requires at least one state_dict")
    if len(state_dicts) != len(sample_counts):
        raise ValueError("state_dicts and sample_counts must have the same length")
    if any(isinstance(count, bool) or not isinstance(count, Integral) or count <= 0
           for count in sample_counts):
        raise ValueError("Each sample count must be a positive integer")
    if any(not isinstance(state, dict) and not hasattr(state, "keys")
           for state in state_dicts):
        raise TypeError("Every state_dict must be a mapping")

    expected_keys = tuple(state_dicts[0].keys())
    if not expected_keys:
        raise ValueError("state_dicts cannot be empty mappings")
    for index, state_dict in enumerate(state_dicts[1:], start=1):
        if set(state_dict.keys()) != set(expected_keys):
            raise ValueError(f"state_dict {index} has a different set of keys")

    for key in expected_keys:
        first = state_dicts[0][key]
        if not isinstance(first, torch.Tensor):
            raise TypeError(f"state_dict value {key!r} must be a tensor")
        for index, state_dict in enumerate(state_dicts[1:], start=1):
            value = state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"state_dict {index} value {key!r} must be a tensor")
            if value.shape != first.shape:
                raise ValueError(f"state_dict tensor shape mismatch for key {key!r}")
            if value.dtype != first.dtype:
                raise ValueError(f"state_dict tensor dtype mismatch for key {key!r}")

    total = float(sum(sample_counts))
    result = OrderedDict()
    for key in expected_keys:
        first = state_dicts[0][key]
        if torch.is_floating_point(first):
            result[key] = sum(
                state_dict[key].float() * (count / total)
                for state_dict, count in zip(state_dicts, sample_counts)
            ).to(first.dtype)
        else:
            # Match existing FedAvg behavior for integer/bool state buffers.
            result[key] = first.clone()
    return result
