"""Safe bounded TCP transport used by S3TO network services.

The framing/codec is JSON with a strict tensor allowlist; it never deserializes
untrusted pickle opcodes. Keep legacy pickle transport out of network-facing code.
"""
from network.secure_tcp import ThreadedTCPServer, send_tcp_message

__all__ = ["ThreadedTCPServer", "send_tcp_message"]
