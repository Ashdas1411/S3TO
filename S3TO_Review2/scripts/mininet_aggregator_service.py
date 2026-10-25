#!/usr/bin/env python3
import argparse
import time
import threading
from collections import defaultdict

from fl.fedavg import fedavg
from network.secure_tcp import ThreadedTCPServer, send_tcp_message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("edge", "global"))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--name", default="edge")
    parser.add_argument("--global-ip", default="10.0.0.103")
    parser.add_argument("--global-port", type=int, default=9200)
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--expected-clients", type=int, default=3)
    parser.add_argument(
        "--allowed-client-ids", default="",
        help="Comma-separated client IDs permitted at this edge service.",
    )
    parser.add_argument(
        "--allow-nonmininet-peers", action="store_true",
        help="Test-only override for localhost protocol tests; never use for runs.",
    )
    parser.add_argument(
        "--aggregation-mode", choices=("hierarchical", "direct"),
        default="hierarchical",
        help="Use hierarchical edge aggregation or direct client-to-global FedAvg.",
    )
    args = parser.parse_args()
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be in the range 1..65535")
    if args.expected_clients < 1:
        parser.error("--expected-clients must be positive")
    if args.role == "edge" and args.aggregation_mode != "hierarchical":
        parser.error("edge services only support hierarchical aggregation")
    try:
        allowed_client_ids = {
            int(value.strip()) for value in args.allowed_client_ids.split(",")
            if value.strip()
        }
    except ValueError:
        parser.error("--allowed-client-ids must be comma-separated integers")
    if any(cid < 0 or cid > 5 for cid in allowed_client_ids):
        parser.error("allowed client IDs must be in the range 0..5")
    if args.role == "edge" and not args.allow_nonmininet_peers:
        if len(allowed_client_ids) != args.expected_clients:
            parser.error(
                "edge services require exactly --expected-clients allowed client IDs"
            )

    lock = threading.Lock()
    edge_updates = defaultdict(list)
    global_updates = defaultdict(dict)

    if args.role == "edge":
        def handle(msg, addr):
            kind = msg.get("type", "model_update")
            round_no = int(msg.get("round", 0))

            if kind == "reset_round":
                if round_no < 1:
                    return {"ok": False, "error": "round must be positive"}
                with lock:
                    edge_updates[round_no] = []
                return {"ok": True, "round": round_no, "reset": True}

            if kind == "model_update":
                cid = int(msg["client_id"])
                sample_count = int(msg["sample_count"])
                state_dict = msg.get("state_dict")
                if round_no < 1 or not 0 <= cid < 6 or sample_count < 1:
                    return {"ok": False, "error": "invalid round, client_id, or sample_count"}
                if not args.allow_nonmininet_peers and addr[0] != f"10.0.0.{11 + cid}":
                    return {"ok": False, "error": "client_id does not match the TCP peer address"}
                if allowed_client_ids and cid not in allowed_client_ids:
                    return {"ok": False, "error": "client is not assigned to this edge"}
                if not isinstance(state_dict, dict) or not state_dict:
                    return {"ok": False, "error": "state_dict must be a non-empty mapping"}
                with lock:
                    current = edge_updates[round_no]
                    if any(item[2] == cid for item in current):
                        return {"ok": False, "error": f"duplicate client {cid}"}
                    if len(current) >= args.expected_clients:
                        return {"ok": False, "error": "expected client update count already reached"}
                    current.append((state_dict, sample_count, cid))
                    count = len(current)

                return {
                    "ok": True,
                    "edge": args.name,
                    "received": count,
                    "round": round_no,
                }

            if kind == "aggregate_request":
                with lock:
                    items = list(edge_updates[round_no])

                if len(items) != args.expected_clients:
                    return {
                        "ok": False,
                        "error": (
                            f"{args.name} has {len(items)} updates; "
                            f"expected {args.expected_clients}"
                        ),
                    }

                state = fedavg(
                    [item[0] for item in items],
                    [item[1] for item in items],
                )
                sample_count = sum(item[1] for item in items)

                response = send_tcp_message(
                    args.global_ip,
                    args.global_port,
                    {
                        "type": "edge_update",
                        "round": round_no,
                        "edge": args.name,
                        "sample_count": sample_count,
                        "state_dict": state,
                    },
                    timeout=120,
                )
                if not response.get("ok"):
                    return {"ok": False, "error": str(response)}

                return {
                    "ok": True,
                    "edge": args.name,
                    "round": round_no,
                    "sample_count": sample_count,
                    "clients_aggregated": len(items),
                    "global_ack": True,
                }

            return {"ok": False, "error": f"unknown message type: {kind}"}

    else:
        def handle(msg, addr):
            kind = msg.get("type", "")
            try:
                round_no = int(msg.get("round", 0))
            except (TypeError, ValueError):
                return {"ok": False, "error": "round must be an integer"}

            if kind == "reset_round":
                if round_no < 1:
                    return {"ok": False, "error": "round must be positive"}
                with lock:
                    global_updates[round_no] = {}
                return {"ok": True, "round": round_no, "reset": True}

            if args.aggregation_mode == "direct" and kind == "model_update":
                try:
                    cid = int(msg["client_id"])
                    sample_count = int(msg["sample_count"])
                except (KeyError, TypeError, ValueError):
                    return {"ok": False, "error": "client_id and sample_count must be integers"}
                state_dict = msg.get("state_dict")
                if round_no < 1 or not 0 <= cid < args.expected_clients or sample_count < 1:
                    return {"ok": False, "error": "invalid round, client_id, or sample_count"}
                if not args.allow_nonmininet_peers and addr[0] != f"10.0.0.{11 + cid}":
                    return {"ok": False, "error": "client_id does not match the TCP peer address"}
                if not isinstance(state_dict, dict) or not state_dict:
                    return {"ok": False, "error": "state_dict must be a non-empty mapping"}
                with lock:
                    updates = global_updates[round_no]
                    if cid in updates:
                        return {"ok": False, "error": f"duplicate client {cid}"}
                    if len(updates) >= args.expected_clients:
                        return {"ok": False, "error": "expected client update count already reached"}
                    updates[cid] = (state_dict, sample_count)
                    count = len(updates)
                return {"ok": True, "received_clients": count, "round": round_no}

            if args.aggregation_mode == "hierarchical" and kind == "edge_update":
                edge = str(msg.get("edge", ""))
                sample_count = msg.get("sample_count")
                state_dict = msg.get("state_dict")
                if round_no < 1 or edge not in {"edge1", "edge2"}:
                    return {"ok": False, "error": "invalid round or edge name"}
                try:
                    sample_count = int(sample_count)
                except (TypeError, ValueError):
                    return {"ok": False, "error": "sample_count must be an integer"}
                if sample_count < 1 or not isinstance(state_dict, dict) or not state_dict:
                    return {"ok": False, "error": "invalid sample_count or state_dict"}
                expected_peer = {"edge1": "10.0.0.101", "edge2": "10.0.0.102"}[edge]
                if not args.allow_nonmininet_peers and addr[0] != expected_peer:
                    return {"ok": False, "error": "edge name does not match the TCP peer address"}
                with lock:
                    updates = global_updates[round_no]
                    if edge in updates:
                        return {"ok": False, "error": f"duplicate edge update {edge}"}
                    if len(updates) >= 2:
                        return {"ok": False, "error": "expected edge update count already reached"}
                    updates[edge] = (state_dict, sample_count)
                    count = len(updates)
                return {"ok": True, "received_edges": count, "round": round_no}

            if kind == "get_global":
                with lock:
                    items = dict(global_updates[round_no])
                expected = args.expected_clients if args.aggregation_mode == "direct" else 2
                if len(items) != expected:
                    unit = "client" if args.aggregation_mode == "direct" else "edge"
                    return {
                        "ok": False,
                        "error": f"global aggregator has {len(items)} {unit} updates; expected {expected}",
                    }
                ordered = [items[key] for key in sorted(items)]
                state = fedavg(
                    [item[0] for item in ordered],
                    [item[1] for item in ordered],
                )
                return {
                    "ok": True,
                    "round": round_no,
                    "sample_count": sum(item[1] for item in ordered),
                    "state_dict": state,
                    "aggregation_mode": args.aggregation_mode,
                    "updates_aggregated": len(items),
                }

            return {"ok": False, "error": f"unknown message type: {kind}"}

    # Path A passes the specific Mininet interface address. Loopback is the
    # safe default if somebody starts this script manually without the runner.
    server = ThreadedTCPServer(args.bind_host, args.port, handle)
    server.start()
    print(f"SERVICE_READY {args.role} {args.port}", flush=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
