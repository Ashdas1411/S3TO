import io
import pickle
import struct

import torch


def state_dict_to_bytes(state_dict):
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


def bytes_to_state_dict(payload):
    buffer = io.BytesIO(payload)
    return torch.load(buffer, map_location="cpu", weights_only=True)


def pack_message(obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return struct.pack("!I", len(payload)) + payload


def recv_exact(sock, n):
    chunks = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed before the full message arrived")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock):
    header = recv_exact(sock, 4)
    size = struct.unpack("!I", header)[0]
    payload = recv_exact(sock, size)
    return pickle.loads(payload)


def send_message(sock, obj):
    sock.sendall(pack_message(obj))
