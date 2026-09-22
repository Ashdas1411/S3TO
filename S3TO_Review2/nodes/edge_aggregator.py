from collections import defaultdict
import threading

from fl.fedavg import fedavg
from network.tcp import ThreadedTCPServer


class EdgeAggregator:
    def __init__(self, name, host, port, global_host, global_port, expected_clients=3):
        self.name = name
        self.host = host
        self.port = port
        self.global_host = global_host
        self.global_port = global_port
        self.expected_clients = expected_clients
        self.round_updates = defaultdict(list)
        self.server = ThreadedTCPServer(host, port, self._handle)
        self.lock = threading.Lock()

    def start(self):
        self.server.start()
        print(f"[EDGE] {self.name} listening on {self.host}:{self.port}")

    def stop(self):
        self.server.stop()

    def reset_round(self, round_no):
        with self.lock:
            self.round_updates[int(round_no)] = []

    def _handle(self, message, addr):
        round_no = int(message["round"])
        with self.lock:
            self.round_updates[round_no].append(
                (message["state_dict"], int(message["sample_count"]), int(message["client_id"]))
            )
        return {"ok": True, "edge": self.name, "received": len(self.round_updates[round_no])}

    def aggregate_round(self, round_no):
        with self.lock:
            items = list(self.round_updates[round_no])
        if len(items) != self.expected_clients:
            raise RuntimeError(
                f"{self.name} has {len(items)} client updates; expected {self.expected_clients}"
            )
        states = [x[0] for x in items]
        counts = [x[1] for x in items]
        return fedavg(states, counts), sum(counts)
