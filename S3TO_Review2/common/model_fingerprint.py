"""Deterministic SHA-256 fingerprint for a PyTorch model state_dict."""
import hashlib
import json

import torch


def state_dict_sha256(state_dict):
    """Hash tensor names, shapes, dtypes, and exact CPU tensor bytes."""
    if not isinstance(state_dict, dict) and not hasattr(state_dict, "items"):
        raise TypeError("model state must be a mapping")
    keys = list(state_dict.keys())
    if any(not isinstance(key, str) for key in keys):
        raise TypeError("model state keys must be strings")
    digest = hashlib.sha256()
    for key in sorted(keys):
        value = state_dict[key]
        if not torch.is_tensor(value):
            raise TypeError("model state must map string keys to tensors")
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError(f"unsupported tensor layout for {key!r}")
        tensor = value.detach().to(device="cpu").contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()
