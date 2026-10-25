#!/usr/bin/env python3
import argparse
import json
import os
import select
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Permit the virtual environment's Python to import Ubuntu's Mininet package.
sys.path.insert(0, "/usr/lib/python3/dist-packages")

import torch

from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.topo import Topo
from mininet.log import setLogLevel

from common.config import load_config
from common.utils import set_seed, save_json
from common.model_fingerprint import state_dict_sha256
from common.data_fingerprint import partition_sha256_by_client
from fl.data import (
    load_mnist, partition_noniid, label_histogram,
    make_loader, make_test_loader,
)
from fl.local_trainer import train_local, evaluate
from fl.model_factory import create_model
from network.telemetry import ClientTelemetry, send_telemetry
from network.openflow_metrics import parse_openflow_flows, summarize_pathA_metrics


ROOT = Path(__file__).resolve().parent
PYTHON = "/home/arkad/s3to/.venv/bin/python"
SERVICE = ROOT / "scripts" / "mininet_aggregator_service.py"


class S3TOMininetTopo(Topo):
    def build(self):
        s1 = self.addSwitch("s1", dpid="0000000000000001")
        s2 = self.addSwitch("s2", dpid="0000000000000002")
        s3 = self.addSwitch("s3", dpid="0000000000000003")

        clients = [
            self.addHost(f"c{i}", ip=f"10.0.0.{10+i}/24")
            for i in range(1, 7)
        ]
        edge1 = self.addHost("edge1", ip="10.0.0.101/24")
        edge2 = self.addHost("edge2", ip="10.0.0.102/24")
        global_agg = self.addHost("gagg", ip="10.0.0.103/24")

        # Keep all existing switch port numbers fixed.
        self.addLink(clients[0], s1, port2=1)
        self.addLink(clients[1], s1, port2=2)
        self.addLink(s1, s2, port1=3, port2=3)
        self.addLink(edge1, s1, port2=4)

        self.addLink(clients[2], s2, port2=1)
        self.addLink(clients[3], s2, port2=2)
        self.addLink(s2, s3, port1=4, port2=4)
        self.addLink(global_agg, s2, port2=5)

        self.addLink(clients[4], s3, port2=1)
        self.addLink(clients[5], s3, port2=2)
        self.addLink(edge2, s3, port2=3)


def controller_command(host, port, message, expected_status):
    """Send one bounded local UDP control command and verify its response."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(5)
        sock.sendto(json.dumps(message).encode("utf-8"), (host, port))
        response, _ = sock.recvfrom(65535)
    result = json.loads(response.decode("utf-8"))
    if result.get("status") != expected_status:
        raise RuntimeError(f"Controller command failed: {result}")
    return result


def reset_ryu(host, port):
    return controller_command(host, port, {"type": "reset"}, "reset")


def apply_direct_mode(host, port):
    return controller_command(
        host, port, {"type": "apply_direct"}, "direct_flows_installed"
    )


def start_service(
    host, role, port, name=None, aggregation_mode="hierarchical", expected_clients=3,
    bind_host="127.0.0.1", allowed_client_ids=None,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    command = [
        PYTHON, "-u", str(SERVICE), role,
        "--port", str(port),
        "--aggregation-mode", aggregation_mode,
        "--expected-clients", str(expected_clients),
        "--bind-host", bind_host,
    ]
    if name:
        command += ["--name", name]
    if allowed_client_ids is not None:
        command += [
            "--allowed-client-ids",
            ",".join(str(cid) for cid in sorted(allowed_client_ids)),
        ]
    if role == "edge":
        command += ["--global-ip", "10.0.0.103", "--global-port", "9200"]

    proc = host.popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    deadline = time.time() + 20
    captured = []
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"{role} service exited: " + "\n".join(captured)
            )
        ready, _, _ = select.select([proc.stdout], [], [], 0.25)
        if ready:
            line = proc.stdout.readline()
            captured.append(line.rstrip())
            if "SERVICE_READY" in line:
                return proc

    proc.terminate()
    raise TimeoutError(
        f"{role} service did not start: " + "\n".join(captured)
    )


def host_rpc(node, host, port, message, state_file=None, state_response=None):
    code = r'''
import json, sys, torch
from network.secure_tcp import send_tcp_message

host, port = sys.argv[1], int(sys.argv[2])
message = json.loads(sys.argv[3])
state_file = sys.argv[4]
state_response = sys.argv[5]

if state_file:
    message["state_dict"] = torch.load(
        state_file, map_location="cpu", weights_only=True
    )

response = send_tcp_message(host, port, message, timeout=180)

if state_response and "state_dict" in response:
    torch.save(response["state_dict"], state_response)
    response.pop("state_dict")

print(json.dumps(response))
'''
    args = [
        PYTHON, "-c", code,
        host, str(port), json.dumps(message),
        str(state_file or ""),
        str(state_response or ""),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    proc = node.popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    output, _ = proc.communicate(timeout=240)

    if proc.returncode != 0:
        raise RuntimeError(
            f"RPC from {node.name} to {host}:{port} failed:\n{output}"
        )

    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Empty RPC response from {node.name}: {output}")

    response = json.loads(lines[-1])
    if not response.get("ok"):
        raise RuntimeError(
            f"RPC rejected by {host}:{port}: {response}"
        )
    return response


def run(args):
    if args.controller_ip not in {"127.0.0.1", "localhost"}:
        raise ValueError(
            "The Path A controller command API is loopback-only for security; "
            "use --controller-ip 127.0.0.1 and run the controller in this WSL instance."
        )
    if args.telemetry_port != 9999:
        raise ValueError(
            "The patched OS-Ken control API is bound to UDP port 9999; "
            "use --telemetry-port 9999."
        )
    cfg = load_config(args.config)
    if int(cfg["fl"]["num_clients"]) != 6:
        raise ValueError("This fixed Mininet topology requires exactly six clients")
    if int(cfg["fl"].get("num_edge_aggregators", 2)) != 2:
        raise ValueError("This fixed Mininet topology requires exactly two edge aggregators")
    if int(cfg["s3to"].get("group_size", 3)) != 3:
        raise ValueError("This fixed S3TO topology requires three clients per edge")
    rounds = int(args.rounds if args.rounds is not None else cfg["fl"]["rounds"])
    if rounds < 1:
        raise ValueError("rounds must be a positive integer")
    set_seed(cfg["fl"]["seed"])

    device = args.device or cfg["fl"].get("device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this environment.")

    data_dir = str(Path(args.data_dir or cfg["fl"]["data_dir"]).expanduser())
    train_ds, test_ds, fallback_used = load_mnist(
        data_dir,
        download=not args.no_download,
        allow_fallback=not args.require_real_mnist,
    )
    if args.require_real_mnist and fallback_used:
        raise RuntimeError("Synthetic fallback data was used; real MNIST required.")

    partitions = partition_noniid(
        train_ds,
        num_clients=cfg["fl"]["num_clients"],
        samples_per_client=cfg["fl"]["samples_per_client"],
        seed=cfg["fl"]["seed"],
        mode=cfg["fl"].get("partition_mode", "fixed"),
        dirichlet_alpha=float(cfg["fl"].get("dirichlet_alpha", 0.5)),
    )
    # label_histogram returns float64 for similarity math elsewhere, but the
    # controller wire schema intentionally accepts integer sample counts only.
    histograms = {
        cid: [int(value) for value in label_histogram(train_ds, indices).tolist()]
        for cid, indices in partitions.items()
    }
    partition_fingerprints = partition_sha256_by_client(train_ds, partitions)

    print("\n=== S3TO REAL CLIENT HISTOGRAMS ===")
    for cid in sorted(histograms):
        print(f"Client {cid+1}: {histograms[cid]}")

    net = Mininet(
        topo=S3TOMininetTopo(),
        controller=None,
        switch=OVSKernelSwitch,
        autoSetMacs=True,
        autoStaticArp=True,
    )
    net.addController(
        "c0",
        controller=RemoteController,
        ip=args.controller_ip,
        port=args.controller_port,
    )

    processes = []
    # A private 0700 directory prevents other local users from reading model
    # checkpoints exchanged with the Mininet host namespaces.
    private_temp_dir = tempfile.TemporaryDirectory(prefix="s3to_mininet_pathA_")
    private_tmp = Path(private_temp_dir.name)
    try:
        net.start()
        if hasattr(net, "waitConnected"):
            connected = net.waitConnected(timeout=15)
            if connected is False:
                raise RuntimeError(
                    "One or more OpenFlow switches did not connect to the controller "
                    "within 15 seconds. Check OS-Ken and port 6653."
                )
        else:
            time.sleep(2)
        # The TCP connection can be up just before the controller has processed
        # switch-features events and registered all three datapaths.
        time.sleep(1)

        edge1, edge2, global_node = net.get("edge1"), net.get("edge2"), net.get("gagg")
        processes.append(start_service(
            global_node,
            "global",
            9200,
            aggregation_mode=args.aggregation_mode,
            expected_clients=cfg["fl"]["num_clients"],
            bind_host="10.0.0.103",
        ))

        reset_ryu(args.controller_ip, args.telemetry_port)
        if args.aggregation_mode == "direct":
            apply_direct_mode(args.controller_ip, args.telemetry_port)
            assignment = {cid: "global" for cid in range(cfg["fl"]["num_clients"])}
            print("\n=== DIRECT FEDAVG PATH ===")
            print("All clients -> global aggregator; edge aggregation bypassed.")
        else:
            if args.assignment_override:
                edges = [item.strip() for item in args.assignment_override.split(",")]
                if (
                    len(edges) != cfg["fl"]["num_clients"]
                    or any(edge not in ("edge1", "edge2") for edge in edges)
                ):
                    raise ValueError(
                        "--assignment-override must provide one edge1/edge2 value "
                        "per client, in client-ID order"
                    )

                assignment = {cid: edge for cid, edge in enumerate(edges)}
                edge_counts = {
                    edge: sum(value == edge for value in assignment.values())
                    for edge in ("edge1", "edge2")
                }
                if edge_counts != {"edge1": 3, "edge2": 3}:
                    raise ValueError(
                        "--assignment-override must assign exactly three clients "
                        "to each edge"
                    )

                response = controller_command(
                    args.controller_ip,
                    args.telemetry_port,
                    {
                        "type": "apply_assignment",
                        "assignment": {
                            str(cid): edge for cid, edge in assignment.items()
                        },
                    },
                    "flows_installed",
                )
                assignment = {
                    int(cid): edge
                    for cid, edge in response["assignment"].items()
                }
                print("\\n=== PATH B ASSIGNMENT OVERRIDE ===")
                for cid in sorted(assignment):
                    print(f"Client {cid+1} -> {assignment[cid]}")
            else:
                assignment = None
                for cid in range(cfg["fl"]["num_clients"]):
                    response = send_telemetry(
                        args.controller_ip,
                        args.telemetry_port,
                        ClientTelemetry(
                            client_id=cid,
                            histogram=histograms[cid],
                            client_ip=f"10.0.0.{11+cid}",
                            model_port=0,
                        ),
                    )
                    assignment = response.get("assignment") or assignment
                    print(
                        f"[UDP] Client {cid+1}: {response.get('status')} "
                        f"({response.get('received_clients')}/"
                        f"{cfg['s3to']['expected_clients']})"
                    )

            if not assignment:
                raise RuntimeError("Controller did not return an S3TO assignment.")
            assignment = {int(cid): edge for cid, edge in assignment.items()}
            edge_counts = {
                edge: sum(value == edge for value in assignment.values())
                for edge in ("edge1", "edge2")
            }
            if edge_counts != {"edge1": 3, "edge2": 3}:
                raise RuntimeError(f"Expected three clients per edge, got {edge_counts}")

            print("\n=== S3TO ASSIGNMENT ===")
            for cid in sorted(assignment):
                print(f"Client {cid+1} -> {assignment[cid]}")

            # Start edge listeners only after assignment is fixed, so each one
            # can enforce the exact client IDs allowed to upload to that edge.
            edge_hosts = {"edge1": edge1, "edge2": edge2}
            edge_ports = {"edge1": 9101, "edge2": 9102}
            for edge_name in ("edge1", "edge2"):
                allowed_ids = [
                    cid for cid, assigned_edge in assignment.items()
                    if assigned_edge == edge_name
                ]
                if len(allowed_ids) != 3:
                    raise RuntimeError(
                        f"{edge_name} must receive three clients, got {allowed_ids}"
                    )
                processes.append(start_service(
                    edge_hosts[edge_name], "edge", edge_ports[edge_name], edge_name,
                    "hierarchical", expected_clients=3,
                    bind_host="10.0.0.101" if edge_name == "edge1" else "10.0.0.102",
                    allowed_client_ids=allowed_ids,
                ))
        time.sleep(1)

        edge_config = {
            "edge1": ("10.0.0.101", 9101),
            "edge2": ("10.0.0.102", 9102),
        }
        net_clients = {cid: net.get(f"c{cid+1}") for cid in assignment}
        results = []
        # Re-seed immediately before model creation so the two modes start from
        # identical weights even if setup internals consume random numbers.
        set_seed(cfg["fl"]["seed"])
        model_global = create_model(args.model).state_dict()
        initial_model_sha256 = state_dict_sha256(model_global)
        print(f"[MODEL] Initial state SHA-256: {initial_model_sha256}")
        test_loader = make_test_loader(test_ds)

        for round_no in range(1, rounds + 1):
            print(f"\n=== MININET FL ROUND {round_no} ===")
            start = time.time()

            # Hierarchical mode resets both edge buffers; direct mode has no
            # edge services and therefore makes no edge reset RPCs.
            if args.aggregation_mode == "hierarchical":
                for edge_name, (edge_ip, port) in edge_config.items():
                    sender_id = next(
                        cid for cid, edge in assignment.items() if edge == edge_name
                    )
                    host_rpc(
                        net_clients[sender_id], edge_ip, port,
                        {"type": "reset_round", "round": round_no},
                    )

            host_rpc(
                global_node, "10.0.0.103", 9200,
                {"type": "reset_round", "round": round_no},
            )

            # Each client trains locally, then transmits its actual weights
            # from its Mininet namespace to its assigned edge server.
            for cid in sorted(assignment):
                model = create_model(args.model)
                model.load_state_dict(model_global)
                loader = make_loader(
                    train_ds,
                    partitions[cid],
                    batch_size=cfg["fl"]["batch_size"],
                    shuffle=True,
                )
                state, loss = train_local(
                    model,
                    loader,
                    epochs=cfg["fl"]["local_epochs"],
                    lr=cfg["fl"]["learning_rate"],
                    device=device,
                )
                state = {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in state.items()
                }

                state_path = private_tmp / f"client_r{round_no}_c{cid}.pt"
                torch.save(state, state_path)

                if args.aggregation_mode == "hierarchical":
                    edge_name = assignment[cid]
                    dest_ip, dest_port = edge_config[edge_name]
                    dest_label = edge_name
                else:
                    dest_ip, dest_port = "10.0.0.103", 9200
                    dest_label = "global"
                response = host_rpc(
                    net_clients[cid],
                    dest_ip,
                    dest_port,
                    {
                        "type": "model_update",
                        "round": round_no,
                        "client_id": cid,
                        "sample_count": len(partitions[cid]),
                    },
                    state_file=state_path,
                )
                receive_count = (
                    response.get("received")
                    if args.aggregation_mode == "hierarchical"
                    else response.get("received_clients")
                )
                print(
                    f"[TCP/Mininet] Client {cid+1} -> {dest_label} | "
                    f"loss={loss:.4f} | received_updates={receive_count}"
                )

            # In S3TO, each edge aggregates its three updates before forwarding
            # one weighted update to the global aggregator. Direct FedAvg skips
            # this stage and sends all six client updates to the global node.
            if args.aggregation_mode == "hierarchical":
                for edge_name, (edge_ip, port) in edge_config.items():
                    sender_id = next(
                        cid for cid, edge in assignment.items() if edge == edge_name
                    )
                    result = host_rpc(
                        net_clients[sender_id],
                        edge_ip,
                        port,
                        {"type": "aggregate_request", "round": round_no},
                    )
                    print(
                        f"[EDGE] {edge_name} aggregated "
                        f"{result['clients_aggregated']} clients; "
                        f"sent {result['sample_count']} samples to global"
                    )

            global_path = private_tmp / f"global_r{round_no}.pt"
            response = host_rpc(
                global_node,
                "10.0.0.103",
                9200,
                {"type": "get_global", "round": round_no},
                state_response=global_path,
            )
            model_global = torch.load(
                global_path, map_location="cpu", weights_only=True
            )
            expected_samples = sum(len(partitions[cid]) for cid in partitions)
            if response.get("sample_count") != expected_samples:
                raise RuntimeError(
                    f"Global aggregated {response.get('sample_count')} samples; "
                    f"expected {expected_samples}"
                )

            model = create_model(args.model)
            model.load_state_dict(model_global)
            accuracy = evaluate(model, test_loader, device)
            elapsed = time.time() - start

            print(
                f"[GLOBAL] Round {round_no}: accuracy={accuracy*100:.2f}% | "
                f"aggregated_samples={response['sample_count']} | "
                f"time={elapsed:.2f}s"
            )
            results.append({
                "round": round_no,
                "accuracy": accuracy,
                "wall_clock_sec": elapsed,
                "aggregated_samples": response["sample_count"],
                "assignment": assignment,
                "aggregation_mode": args.aggregation_mode,
            })

        # Capture switch counters before Mininet tears down the topology.
        flow_stats = {}
        parsed_flows = {}
        for switch_name in ("s1", "s2", "s3"):
            completed = subprocess.run(
                ["ovs-ofctl", "-O", "OpenFlow13", "dump-flows", switch_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            raw = completed.stdout
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Failed to read OpenFlow counters for {switch_name}: "
                    f"{completed.stderr.strip()}"
                )
            rows = parse_openflow_flows(raw)
            parsed_flows[switch_name] = rows
            packet_total = sum(row["packets"] for row in rows)
            byte_total = sum(row["bytes"] for row in rows)

            flow_stats[switch_name] = {
                "returncode": completed.returncode,
                "parsed_flow_count": len(rows),
                "packets_total": packet_total,
                "bytes_total": byte_total,
                "raw_flows": raw,
                "stderr": completed.stderr,
            }
            print(
                f"[OpenFlow] {switch_name}: packets={packet_total}, "
                f"bytes={byte_total}, command_rc={completed.returncode}"
            )

        network_metrics = summarize_pathA_metrics(parsed_flows, args.aggregation_mode)
        if args.aggregation_mode == "hierarchical":
            default_output = f"results/s3to_mininet_{args.model}_real_mnist.json"
            transport = "Mininet TCP client-to-edge and edge-to-global using bounded JSON/tensor framing"
        else:
            default_output = f"results/pathA_direct_{args.model}_real_mnist.json"
            transport = "Mininet TCP client-to-global direct FedAvg using bounded JSON/tensor framing"
        output_path = Path(args.output or default_output)
        save_json({
            "experiment": "S3TO Path A network comparison",
            "aggregation_mode": args.aggregation_mode,
            "model": args.model,
            "dataset": "real_mnist" if not fallback_used else "synthetic_digit_like",
            "config": {
                "seed": cfg["fl"]["seed"],
                "partition_mode": cfg["fl"].get("partition_mode", "fixed"),
                "dirichlet_alpha": cfg["fl"].get("dirichlet_alpha", 0.5),
                "num_clients": cfg["fl"]["num_clients"],
                "samples_per_client": cfg["fl"]["samples_per_client"],
                "local_epochs": cfg["fl"]["local_epochs"],
                "batch_size": cfg["fl"]["batch_size"],
                "learning_rate": cfg["fl"]["learning_rate"],
                "rounds": rounds,
                "device": device,
            },
            "assignment": assignment,
            "histograms": histograms,
            "partition_sha256_by_client": partition_fingerprints,
            "data_summary": {
                "full_training_examples_available": len(train_ds),
                "test_examples": len(test_ds),
                "federated_training_examples_per_round": sum(
                    len(indices) for indices in partitions.values()
                ),
            },
            "initial_model_sha256": initial_model_sha256,
            "results": results,
            "openflow_counters": flow_stats,
            "network_metrics": network_metrics,
            "transport": transport,
            "note": (
                "Training is orchestrated by the runner, but all model-update "
                "messages traverse Mininet host namespaces. The s2 port-5 "
                "counters uniquely measure traffic crossing the global host link. "
                "Summed per-switch byte counters are byte-hops and are not unique bytes."
            ),
        }, output_path)
        print(f"\nSaved results: {output_path}")
        print(f"Aggregation mode: {args.aggregation_mode}")
        print("Mininet client and aggregator model-update traffic completed.")

    finally:
        try:
            for proc in processes:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    pass
            net.stop()
        finally:
            private_temp_dir.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--model", choices=("mlp", "cnn"), default="mlp")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--require-real-mnist", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--aggregation-mode", choices=("hierarchical", "direct"),
        default="hierarchical",
        help="hierarchical runs S3TO; direct bypasses edge aggregation for Path A.",
    )
    parser.add_argument("--assignment-override", default=None, help="Optional six-item edge1/edge2 mapping for controlled Path B experiments.")
    parser.add_argument("--controller-ip", default="127.0.0.1")
    parser.add_argument("--controller-port", type=int, default=6653)
    parser.add_argument("--telemetry-port", type=int, default=9999)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.assignment_override and args.aggregation_mode != "hierarchical":
        parser.error("--assignment-override is only valid in hierarchical mode")
    run(args)
