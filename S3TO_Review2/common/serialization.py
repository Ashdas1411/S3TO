"""Safe message framing and weights-only state-dict serialization."""
import io

import torch
from network.secure_serialization import (
    pack_message, recv_exact, recv_message, send_message,
)


def state_dict_to_bytes(state_dict):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def bytes_to_state_dict(payload):
    buffer = io.BytesIO(payload)
    return torch.load(buffer, map_location="cpu", weights_only=True)

__all__ = [
    "state_dict_to_bytes", "bytes_to_state_dict", "pack_message",
    "recv_exact", "recv_message", "send_message",
]
