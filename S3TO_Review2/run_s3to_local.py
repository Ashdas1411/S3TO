import argparse
import json
import time
import socket
from pathlib import Path

import torch

from common.config import load_config
from common.serialization import state_dict_to_bytes, bytes_to_state_dict
from common.utils import set_seed, save_json
from fl.data import load_mnist, partition_noniid, label_histogram, make_loader, make_test_loader
from fl.local_trainer import train_local, evaluate
from fl.model_factory import create_model
from network.local_controller import LocalS3TOController
from network.telemetry import ClientTelemetry, send_telemetry
from network.tcp import send_tcp_message
from nodes.edge_aggregator import EdgeAggregator
from nodes.global_aggregator import GlobalAggregator
from network.simulated_paths import path_for


def reset_remote_controller(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(5)
        sock.sendto(json.dumps({"type": "reset"}).encode("utf-8"), (host, port))
        data, _ = sock.recvfrom(65535)
    response = json.loads(data.decode("utf-8"))
    if response.get("status") != "reset":
        raise RuntimeError(f"Ryu reset failed: {response}")
    return response


def run(args):
    cfg = load_config(args.config)
    set_seed(cfg["fl"]["seed"])
    device = args.device or cfg["fl"].get("device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    data_dir = str(Path(args.data_dir or cfg["fl"]["data_dir"]).expanduser())
    train_ds, test_ds, used_fallback = load_mnist(
        data_dir,
        download=not args.no_download,
        allow_fallback=not args.require_real_mnist,
    )
    partitions = partition_noniid(
        train_ds,
        num_clients=cfg["fl"]["num_clients"],
        samples_per_client=cfg["fl"]["samples_per_client"],
        seed=cfg["fl"]["seed"],
        mode=cfg["fl"].get("partition_mode", "fixed"),
        dirichlet_alpha=float(cfg["fl"].get("dirichlet_alpha", 0.5)),
    )
    histograms = {
        cid: label_histogram(train_ds, indices).tolist()
        for cid, indices in partitions.items()
    }

    print("\n=== S3TO LABEL DISTRIBUTIONS ===")
    for cid in sorted(histograms):
        print(f"Client {cid+1}: {histograms[cid]}")

    controller = None
    if args.controller == "local":
        controller = LocalS3TOController(
            host=cfg["network"]["telemetry_host"],
            port=cfg["network"]["telemetry_port"],
            expected_clients=cfg["s3to"]["expected_clients"],
            group_size=cfg["s3to"]["group_size"],
        )
        controller.start()

    # In Ryu mode, UDP telemetry goes to the Ryu controller at the
    # configured telemetry address. Do not start a second UDP listener.
    controller_host = (
        controller.host if controller is not None
        else cfg["network"]["telemetry_host"]
    )
    controller_port = (
        controller.port if controller is not None
        else cfg["network"]["telemetry_port"]
    )

    def stop_all():
        if controller is not None:
            controller.stop()
        edge1.stop()
        edge2.stop()
        global_agg.stop()

    edge_ports = cfg["network"]["edge_ports"]
    edge1 = EdgeAggregator("edge1", cfg["network"]["edge_hosts"]["edge1"], edge_ports["edge1"],
                           cfg["network"]["global_host"], cfg["network"]["global_port"], expected_clients=3)
    edge2 = EdgeAggregator("edge2", cfg["network"]["edge_hosts"]["edge2"], edge_ports["edge2"],
                           cfg["network"]["global_host"], cfg["network"]["global_port"], expected_clients=3)
    global_agg = GlobalAggregator(cfg["network"]["global_host"], cfg["network"]["global_port"], expected_edges=2)
    edge1.start(); edge2.start(); global_agg.start()

    try:
        # Clear previous controller state before submitting fresh histograms.
        if args.controller == "ryu":
            reset_remote_controller(controller_host, controller_port)
            print("[UDP] Ryu telemetry and previous assignment flows reset.")

        # Phase A: UDP telemetry -> S3TO decision.
        assignment = None
        for cid in range(cfg["fl"]["num_clients"]):
            response = send_telemetry(
                controller_host,
                controller_port,
                ClientTelemetry(client_id=cid, histogram=histograms[cid], model_port=0),
            )
            assignment = response.get("assignment") or assignment
            print(f"[UDP] Client {cid+1} -> controller | status={response['status']}")

        if not assignment:
            raise RuntimeError("Controller did not produce an assignment")
        assignment = {int(k): v for k, v in assignment.items()}
        print("\n=== S3TO ASSIGNMENT ===")
        for cid in sorted(assignment):
            print(f"Client {cid+1} -> {assignment[cid]} via {path_for(cid, assignment[cid])}")

        model_global = create_model(args.model).state_dict()
        test_loader = make_test_loader(test_ds)
        results = []
        rounds = int(args.rounds or cfg["fl"]["rounds"])

        for round_no in range(1, rounds + 1):
            print(f"\n=== ROUND {round_no} ===")
            edge1.reset_round(round_no)
            edge2.reset_round(round_no)
            model_states = {}
            start = time.time()

            # Each client receives the current global model and trains locally.
            for cid in range(cfg["fl"]["num_clients"]):
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
                model_states[cid] = state
                edge = assignment[cid]
                host = cfg["network"]["edge_hosts"][edge]
                port = cfg["network"]["edge_ports"][edge]
                response = send_tcp_message(
                    host,
                    port,
                    {
                        "type": "model_update",
                        "round": round_no,
                        "client_id": cid,
                        "sample_count": len(partitions[cid]),
                        "state_dict": state,
                    },
                )
                print(f"[TCP] Client {cid+1} -> {edge} | loss={loss:.4f} | ack={response['received']}")

            # Hierarchical aggregation: clients -> edges -> global.
            edge_states = []
            for edge_obj, edge_name in [(edge1, "edge1"), (edge2, "edge2")]:
                edge_state, count = edge_obj.aggregate_round(round_no)
                send_tcp_message(
                    cfg["network"]["global_host"],
                    cfg["network"]["global_port"],
                    {
                        "type": "edge_update",
                        "round": round_no,
                        "edge": edge_name,
                        "sample_count": count,
                        "state_dict": edge_state,
                    },
                )
                edge_states.append((edge_name, edge_state, count))
                print(f"[TCP] {edge_name} -> global | samples={count}")

            model_global = global_agg.aggregate(round_no)
            acc = evaluate(_model_from_state(model_global, args.model), test_loader, device)
            elapsed = time.time() - start
            results.append({"round": round_no, "accuracy": acc, "wall_clock_sec": elapsed})
            print(f"[GLOBAL] Round {round_no} complete | accuracy={acc*100:.2f}% | time={elapsed:.2f}s")

        dataset_tag = "synthetic_digit_like" if used_fallback else "real_mnist"
        output = {
            "model": args.model,
            "dataset": dataset_tag,
            "dataset_fallback_used": used_fallback,
            "data_dir": data_dir,
            "device": device,
            "train_samples": len(train_ds),
            "test_samples": len(test_ds),
            "assignment": assignment,
            "histograms": histograms,
            "results": results,
        }
        default_output = Path("results") / f"s3to_local_{args.model}_{dataset_tag}.json"
        output_path = Path(args.output) if args.output else default_output
        save_json(output, output_path)
        print(f"\nSaved: {output_path}")
    finally:
        stop_all()


def _model_from_state(state, model_name="mlp"):
    model = create_model(model_name)
    model.load_state_dict(state)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--controller",
        choices=("local", "ryu"),
        default="local",
        help="Use the local UDP controller or the running Ryu controller",
    )
    parser.add_argument("--model", choices=("mlp", "cnn"), default="mlp")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--require-real-mnist", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run(args)
