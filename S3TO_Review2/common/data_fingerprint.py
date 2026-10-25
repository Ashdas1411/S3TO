"""Deterministic fingerprints for the exact per-client training examples."""
import hashlib
import struct

import torch


def partition_sha256_by_client(dataset, partitions):
    """Hash each client's ordered sample indices, input tensors, and labels.

    Dataset samples are read exactly as ``__getitem__`` exposes them to the
    training pipeline, after deterministic transforms such as MNIST ToTensor.
    The returned keys are strings so they serialize predictably to JSON.
    """
    fingerprints = {}
    for raw_client_id in sorted(partitions):
        client_id = int(raw_client_id)
        indices = [int(index) for index in partitions[raw_client_id]]
        digest = hashlib.sha256()
        digest.update(b"S3TO-PARTITION-V1\0")
        digest.update(struct.pack("!q", client_id))
        digest.update(struct.pack("!Q", len(indices)))
        for index in indices:
            image, label = dataset[index]
            tensor = torch.as_tensor(image).detach().cpu().contiguous()
            if tensor.layout != torch.strided or tensor.is_quantized:
                raise TypeError("Only dense non-quantized input tensors can be fingerprinted")
            label_value = int(label.item()) if torch.is_tensor(label) else int(label)
            digest.update(struct.pack("!q", index))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b"\0")
            shape_bytes = b",".join(str(int(dim)).encode("ascii") for dim in tensor.shape)
            digest.update(shape_bytes)
            digest.update(b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            digest.update(struct.pack("!q", label_value))
        fingerprints[str(client_id)] = digest.hexdigest()
    return fingerprints
