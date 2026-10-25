#!/usr/bin/env python3
"""Laptop-side receiver for a two-physical-client FL demonstration."""
import argparse
import json
import threading
from pathlib import Path

import torch
from fl.fedavg import fedavg
from network.tcp import ThreadedTCPServer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument(
        "--allowed-client-ips", default="",
        help="Comma-separated Pi IPv4 allowlist; required when binding beyond loopback.",
    )
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--output-dir", default="results/hardware")
    args = parser.parse_args()
    allowed_client_ips = {
        value.strip() for value in args.allowed_client_ips.split(",") if value.strip()
    }
    if args.host not in {"127.0.0.1", "localhost"} and not allowed_client_ips:
        parser.error(
            "non-loopback binding requires --allowed-client-ips; also restrict the "
            "port with the host firewall to those Pi addresses"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    updates = {}
    completed = set()
    receipts = []

    def handle(msg, addr):
        if allowed_client_ips and addr[0] not in allowed_client_ips:
            return {"ok": False, "error": "source IP is not allowlisted"}
        if not isinstance(msg, dict):
            return {"ok": False, "error": "request must be an object"}
        kind = msg.get("type")

        if kind == "ping":
            print(f"[PING] {msg.get('client_id')} from {addr[0]}", flush=True)
            return {"ok": True, "message": "S3TO receiver reachable"}

        if kind != "model_update":
            return {"ok": False, "error": f"Unknown message type: {kind}"}

        round_no = int(msg["round"])
        client_id = str(msg["client_id"])
        count = int(msg["sample_count"])
        state = msg.get("state_dict")
        if not isinstance(state, dict) or count < 1:
            return {"ok": False, "error": "Missing state_dict or invalid sample_count"}

        should_aggregate = False

        with lock:
            if round_no in completed:
                return {"ok": False, "error": f"Round {round_no} already aggregated"}

            bucket = updates.setdefault(round_no, {})
            if client_id in bucket:
                return {"ok": False, "error": f"Duplicate client {client_id} in round {round_no}"}

            bucket[client_id] = (state, count)
            received = len(bucket)
            print(
                f"[UPDATE] round={round_no}, client={client_id}, "
                f"samples={count}, received={received}/{args.clients}",
                flush=True,
            )

            if received == args.clients:
                completed.add(round_no)
                should_aggregate = True
                ordered = [bucket[k] for k in sorted(bucket)]

        aggregated = False
        if should_aggregate:
            global_state = fedavg(
                [item[0] for item in ordered],
                [item[1] for item in ordered],
            )
            target = out / f"physical_global_round_{round_no}.pt"
            torch.save(global_state, target)

            receipt = {
                "round": round_no,
                "client_ids": sorted(updates[round_no]),
                "aggregated_samples": sum(item[1] for item in ordered),
                "checkpoint": str(target),
            }
            with lock:
                receipts.append(receipt)
                (out / "physical_receipts.json").write_text(
                    json.dumps(receipts, indent=2)
                )

            print(f"[FEDAVG] Round {round_no} saved to {target}", flush=True)
            aggregated = True

        return {
            "ok": True,
            "round": round_no,
            "received_clients": received,
            "expected_clients": args.clients,
            "aggregated": aggregated,
        }

    server = ThreadedTCPServer(args.host, args.port, handle)
    server.start()
    print(
        f"SERVER_READY listening={args.host}:{args.port} "
        f"expected_clients={args.clients}",
        flush=True,
    )

    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("Stopping receiver...")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
