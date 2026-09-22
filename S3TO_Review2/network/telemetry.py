import json
import socket
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class ClientTelemetry:
    client_id: int
    histogram: list
    client_ip: str = ""
    model_port: int = 0
    link_quality: dict = field(default_factory=dict)

    def to_json(self):
        return json.dumps(asdict(self)).encode("utf-8")


def send_telemetry(host, port, telemetry: ClientTelemetry, timeout=5):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        sock.sendto(telemetry.to_json(), (host, port))
        data, _ = sock.recvfrom(65535)
        return json.loads(data.decode("utf-8"))


def send_controller_command(host, port, command):
    payload = json.dumps(command).encode("utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(payload, (host, port))
