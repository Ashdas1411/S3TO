"""Portable, non-pickle wire codec for the standalone Raspberry Pi client.

This file intentionally imports PyTorch lazily so --mode smoke remains usable
on a Pi before PyTorch is installed. Its wire format matches
network.secure_serialization used by the receiver.
"""
import base64
import json
import math
import socket
import struct

MAX_MESSAGE_BYTES = 64 * 1024 * 1024
MAX_NESTING_DEPTH = 64
_TENSOR_TAG = "__s3to_tensor_v1__"
_DICT_TAG = "__s3to_dict_v1__"
_BYTES_TAG = "__s3to_bytes_v1__"
_TORCH_CACHE = None
_TORCH_LOOKED_UP = False
_DTYPE_NAMES = (
    "bool", "uint8", "int8", "int16", "int32", "int64", "float16",
    "bfloat16", "float32", "float64",
)


def _torch_module():
    global _TORCH_CACHE, _TORCH_LOOKED_UP
    if _TORCH_LOOKED_UP:
        return _TORCH_CACHE
    _TORCH_LOOKED_UP = True
    try:
        import torch
    except ImportError:
        _TORCH_CACHE = None
    else:
        _TORCH_CACHE = torch
    return _TORCH_CACHE


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def _encode(value, depth=0):
    if depth > MAX_NESTING_DEPTH:
        raise ValueError("Message nesting exceeds the allowed depth")
    torch = _torch_module()
    if torch is not None and isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.is_quantized:
            raise TypeError("Only dense, non-quantized tensors are supported")
        tensor = value.detach().to(device="cpu").contiguous()
        names = {getattr(torch, name): name for name in _DTYPE_NAMES}
        dtype_name = names.get(tensor.dtype)
        if dtype_name is None:
            raise TypeError(f"Unsupported tensor dtype: {tensor.dtype}")
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        if len(raw) > MAX_MESSAGE_BYTES:
            raise ValueError("Tensor exceeds the maximum message size")
        return {_TENSOR_TAG: {
            "dtype": dtype_name,
            "shape": list(tensor.shape),
            "data": base64.b64encode(raw).decode("ascii"),
        }}
    if isinstance(value, bytes):
        if len(value) > MAX_MESSAGE_BYTES:
            raise ValueError("Byte field exceeds the maximum message size")
        return {_BYTES_TAG: base64.b64encode(value).decode("ascii")}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite floating-point values are not allowed")
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Message dictionary keys must be strings")
        return {_DICT_TAG: [[key, _encode(item, depth + 1)] for key, item in value.items()]}
    if isinstance(value, (list, tuple)):
        return [_encode(item, depth + 1) for item in value]
    raise TypeError(f"Unsupported message value type: {type(value).__name__}")


def _decode(value, depth=0):
    if depth > MAX_NESTING_DEPTH:
        raise ValueError("Message nesting exceeds the allowed depth")
    if isinstance(value, dict):
        if set(value) == {_DICT_TAG}:
            pairs = value[_DICT_TAG]
            if not isinstance(pairs, list):
                raise ValueError("Malformed mapping envelope")
            result = {}
            for pair in pairs:
                if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
                    raise ValueError("Malformed mapping entry")
                if pair[0] in result:
                    raise ValueError("Duplicate mapping key")
                result[pair[0]] = _decode(pair[1], depth + 1)
            return result
        if set(value) == {_BYTES_TAG}:
            encoded = value[_BYTES_TAG]
            if not isinstance(encoded, str):
                raise ValueError("Malformed byte envelope")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                raise ValueError("Malformed base64 byte field") from exc
            if len(raw) > MAX_MESSAGE_BYTES:
                raise ValueError("Decoded byte field exceeds the maximum size")
            return raw
        if set(value) == {_TENSOR_TAG}:
            torch = _torch_module()
            if torch is None:
                raise RuntimeError("Tensor messages require PyTorch on the client")
            spec = value[_TENSOR_TAG]
            if not isinstance(spec, dict) or set(spec) != {"dtype", "shape", "data"}:
                raise ValueError("Malformed tensor envelope")
            allowed = {name: getattr(torch, name) for name in _DTYPE_NAMES}
            dtype = allowed.get(spec["dtype"])
            if dtype is None:
                raise ValueError("Tensor dtype is not permitted")
            shape = spec["shape"]
            if not isinstance(shape, list) or len(shape) > MAX_NESTING_DEPTH:
                raise ValueError("Invalid tensor shape")
            if any(type(dim) is not int or dim < 0 for dim in shape):
                raise ValueError("Invalid tensor dimension")
            elements = math.prod(shape) if shape else 1
            expected_bytes = elements * torch.empty((), dtype=dtype).element_size()
            if expected_bytes > MAX_MESSAGE_BYTES:
                raise ValueError("Decoded tensor exceeds the maximum size")
            try:
                raw = base64.b64decode(spec["data"], validate=True)
            except (ValueError, TypeError, base64.binascii.Error) as exc:
                raise ValueError("Malformed base64 tensor data") from exc
            if len(raw) != expected_bytes:
                raise ValueError("Tensor byte length does not match its shape")
            if elements == 0:
                return torch.empty(shape, dtype=dtype)
            return torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(shape)
        raise ValueError("Unknown JSON message envelope")
    if isinstance(value, list):
        return [_decode(item, depth + 1) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite floating-point values are not allowed")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("Unsupported JSON value in message")


def pack_message(obj):
    payload = json.dumps(
        _encode(obj), separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    if not payload or len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Message exceeds the permitted size")
    return struct.pack("!I", len(payload)) + payload


def recv_exact(sock, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed before the full message arrived")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock):
    size = struct.unpack("!I", recv_exact(sock, 4))[0]
    if size < 1 or size > MAX_MESSAGE_BYTES:
        raise ValueError(f"Invalid message size: {size}")
    try:
        parsed = json.loads(
            recv_exact(sock, size).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Message must be valid UTF-8 JSON") from exc
    return _decode(parsed)
