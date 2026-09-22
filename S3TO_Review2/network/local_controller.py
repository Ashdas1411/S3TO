import json
import socket
import threading

from s3to.engine import S3TOEngine


class LocalS3TOController:
    """Local UDP controller used for the fully integrated demo.

    This mirrors the decision logic used by the Ryu app, while avoiding the
    privilege and namespace complexity of Mininet during early development.
    """

    def __init__(self, host="127.0.0.1", port=9999, expected_clients=6, group_size=3):
        self.host = host
        self.port = port
        self.expected_clients = expected_clients
        self.engine = S3TOEngine(group_size=group_size)
        self.telemetry = {}
        self.assignment = None
        self._stop = threading.Event()
        self._thread = None
        self._sock = None
        self.lock = threading.Lock()

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.5)
        self._sock.bind((self.host, self.port))
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
                if msg.get("type") == "reset":
                    with self.lock:
                        self.telemetry.clear()
                        self.assignment = None
                    self._sock.sendto(json.dumps({"status": "reset"}).encode(), addr)
                    continue

                client_id = int(msg["client_id"])
                histogram = msg["histogram"]
                with self.lock:
                    self.telemetry[client_id] = {
                        "histogram": histogram,
                        "client_ip": msg.get("client_ip", ""),
                        "model_port": msg.get("model_port", 0),
                        "link_quality": msg.get("link_quality", {}),
                    }
                    if len(self.telemetry) >= self.expected_clients:
                        result = self.engine.decide({
                            cid: item["histogram"] for cid, item in self.telemetry.items()
                        })
                        self.assignment = result.assignment

                response = {
                    "status": "ready" if self.assignment else "collecting",
                    "received_clients": len(self.telemetry),
                    "assignment": self.assignment or {},
                }
                self._sock.sendto(json.dumps(response).encode("utf-8"), addr)
            except Exception as exc:
                error = {"status": "error", "message": str(exc)}
                self._sock.sendto(json.dumps(error).encode("utf-8"), addr)

    def stop(self):
        self._stop.set()
        if self._sock:
            self._sock.close()
        if self._thread:
            self._thread.join(timeout=1)
