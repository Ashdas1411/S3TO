from network.tcp import ThreadedTCPServer, send_tcp_message
from fl.fedavg import fedavg


class GlobalAggregator:
    def __init__(self, host="127.0.0.1", port=9200, expected_edges=2):
        self.host = host
        self.port = port
        self.expected_edges = expected_edges
        self.server = ThreadedTCPServer(host, port, self._handle)
        self.received = {}

    def start(self):
        self.server.start()
        print(f"[GLOBAL] listening on {self.host}:{self.port}")

    def stop(self):
        self.server.stop()

    def _handle(self, message, addr):
        round_no = int(message["round"])
        self.received.setdefault(round_no, {})[message["edge"]] = (
            message["state_dict"], int(message["sample_count"])
        )
        return {"ok": True, "received_edges": len(self.received[round_no])}

    def aggregate(self, round_no):
        items = list(self.received.get(round_no, {}).values())
        if len(items) != self.expected_edges:
            raise RuntimeError(
                f"Global aggregator has {len(items)} edge updates; expected {self.expected_edges}"
            )
        states = [x[0] for x in items]
        counts = [x[1] for x in items]
        return fedavg(states, counts)
