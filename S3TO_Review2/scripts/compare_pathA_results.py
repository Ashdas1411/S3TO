#!/usr/bin/env python3
"""Compare one direct-FedAvg JSON result with one hierarchical S3TO result."""
import argparse
import json
import math
import re
from pathlib import Path


def _load(path, expected_mode):
    with Path(path).open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("aggregation_mode") != expected_mode:
        raise ValueError(
            f"{path}: expected aggregation_mode={expected_mode!r}, "
            f"found {result.get('aggregation_mode')!r}"
        )
    if result.get("dataset") != "real_mnist":
        raise ValueError(f"{path}: results must use real MNIST")
    if not result.get("results"):
        raise ValueError(f"{path}: result has no rounds")
    if not isinstance(result.get("network_metrics"), dict):
        raise ValueError(f"{path}: network_metrics missing; rerun with Path A code")
    return result


def compare(hierarchical_path, direct_path, accuracy_tolerance_pp=0.10):
    hierarchical = _load(hierarchical_path, "hierarchical")
    direct = _load(direct_path, "direct")
    if hierarchical.get("model") != direct.get("model"):
        raise ValueError("Model architectures differ between runs")
    hc, dc = hierarchical.get("config", {}), direct.get("config", {})
    comparable_keys = (
        "seed", "partition_mode", "dirichlet_alpha", "num_clients",
        "samples_per_client", "local_epochs", "batch_size", "learning_rate",
        "rounds", "device",
    )
    mismatches = {
        key: (hc.get(key), dc.get(key))
        for key in comparable_keys if hc.get(key) != dc.get(key)
    }
    if mismatches:
        raise ValueError(f"Experiment configs differ: {mismatches}")
    if hierarchical.get("histograms") != direct.get("histograms"):
        raise ValueError("Client label histograms differ; comparison is not controlled")
    hpartitions = hierarchical.get("partition_sha256_by_client")
    dpartitions = direct.get("partition_sha256_by_client")
    expected_client_keys = {str(cid) for cid in range(6)}
    if (
        not isinstance(hpartitions, dict)
        or not isinstance(dpartitions, dict)
        or set(hpartitions) != expected_client_keys
        or set(dpartitions) != expected_client_keys
        or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
               for value in list(hpartitions.values()) + list(dpartitions.values()))
    ):
        raise ValueError("Valid exact client data fingerprints missing; rerun both experiments")
    if hpartitions != dpartitions:
        raise ValueError("Per-client training examples differ; comparison is not controlled")
    hhash = hierarchical.get("initial_model_sha256")
    dhash = direct.get("initial_model_sha256")
    if not isinstance(hhash, str) or not isinstance(dhash, str):
        raise ValueError("Initial model fingerprint missing; rerun both experiments")
    if (
        not isinstance(hhash, str)
        or not isinstance(dhash, str)
        or re.fullmatch(r"[0-9a-f]{64}", hhash) is None
        or re.fullmatch(r"[0-9a-f]{64}", dhash) is None
    ):
        raise ValueError("Valid initial model fingerprint missing; rerun both experiments")
    if hhash != dhash:
        raise ValueError("Initial model weights differ; comparison is not controlled")
    hdata, ddata = hierarchical.get("data_summary"), direct.get("data_summary")
    if not isinstance(hdata, dict) or not isinstance(ddata, dict):
        raise ValueError("Dataset summary missing; rerun both experiments")
    if hdata != ddata:
        raise ValueError("Dataset sizes or federated sample totals differ")
    if len(hierarchical["results"]) != len(direct["results"]):
        raise ValueError("The experiments have different numbers of rounds")

    hrounds = hierarchical["results"]
    drounds = direct["results"]
    round_rows = []
    for hr, dr in zip(hrounds, drounds):
        if int(hr["round"]) != int(dr["round"]):
            raise ValueError("Round indices do not align")
        hacc = float(hr["accuracy"]) * 100
        dacc = float(dr["accuracy"]) * 100
        round_rows.append({
            "round": int(hr["round"]),
            "s3to_accuracy_pct": hacc,
            "direct_accuracy_pct": dacc,
            "direct_minus_s3to_pp": dacc - hacc,
            "s3to_wall_clock_sec": float(hr["wall_clock_sec"]),
            "direct_wall_clock_sec": float(dr["wall_clock_sec"]),
        })

    hnetwork = hierarchical["network_metrics"]
    dnetwork = direct["network_metrics"]
    hlink = hnetwork["global_host_link_at_s2_port5"]
    dlink = dnetwork["global_host_link_at_s2_port5"]
    hhop_metrics = hnetwork.get("ipv4_forwarding_egress_byte_hops", {})
    dhop_metrics = dnetwork.get("ipv4_forwarding_egress_byte_hops", {})
    hhop = hhop_metrics.get("all_origins", {})
    dhop = dhop_metrics.get("all_origins", {})
    hclient = hhop_metrics.get("client_originated", {})
    dclient = dhop_metrics.get("client_originated", {})
    hedge = hhop_metrics.get("edge_originated", {})
    dedge = dhop_metrics.get("edge_originated", {})
    if any(not isinstance(item.get("bytes"), int) for item in (hhop, dhop, hclient, dclient, hedge, dedge)):
        raise ValueError("IPv4 forwarding byte-hop metrics missing; rerun both experiments")
    hupload = hlink["uploads_to_global"]
    dupload = dlink["uploads_to_global"]
    hresponses = hlink["responses_from_global"]
    dresponses = dlink["responses_from_global"]
    hseconds = sum(row["wall_clock_sec"] for row in hrounds)
    dseconds = sum(row["wall_clock_sec"] for row in drounds)
    hfinal = float(hrounds[-1]["accuracy"]) * 100
    dfinal = float(drounds[-1]["accuracy"]) * 100
    accuracy_delta = dfinal - hfinal
    upload_reduction_pct = (
        (1 - hupload["bytes"] / dupload["bytes"]) * 100
        if dupload["bytes"] > 0 else None
    )
    return {
        "comparison": "S3TO hierarchical vs direct FedAvg",
        "controlled_config": hc,
        "final_accuracy_pct": {
            "s3to_hierarchical": hfinal,
            "direct_fedavg": dfinal,
            "direct_minus_s3to_percentage_points": accuracy_delta,
            "within_configured_tolerance": abs(accuracy_delta) <= accuracy_tolerance_pp,
            "tolerance_percentage_points": accuracy_tolerance_pp,
        },
        "wall_clock": {
            "s3to_total_round_seconds": hseconds,
            "direct_total_round_seconds": dseconds,
            "direct_minus_s3to_seconds": dseconds - hseconds,
            "s3to_mean_round_seconds": hseconds / len(hrounds),
            "direct_mean_round_seconds": dseconds / len(drounds),
        },
        "global_host_link": {
            "s3to_upload_packets": hupload["packets"],
            "s3to_upload_bytes": hupload["bytes"],
            "direct_upload_packets": dupload["packets"],
            "direct_upload_bytes": dupload["bytes"],
            "s3to_upload_byte_reduction_vs_direct_pct": upload_reduction_pct,
            "s3to_response_packets_from_global": hresponses["packets"],
            "s3to_response_bytes_from_global": hresponses["bytes"],
            "direct_response_packets_from_global": dresponses["packets"],
            "direct_response_bytes_from_global": dresponses["bytes"],
            "interpretation": (
                "OpenFlow forwarding-rule counters track client/edge updates toward the "
                "global host and responses returning from it. This runner does not send "
                "the global model to clients over Mininet; these are not full-FL downlink totals."
            ),
        },
        "network_load_proxy": {
            "s3to_ipv4_forwarding_byte_hops": hhop["bytes"],
            "direct_ipv4_forwarding_byte_hops": dhop["bytes"],
            "s3to_byte_hop_reduction_vs_direct_pct": (
                (1 - hhop["bytes"] / dhop["bytes"]) * 100
                if dhop["bytes"] > 0 else None
            ),
            "s3to_client_originated_byte_hops": hclient["bytes"],
            "direct_client_originated_byte_hops": dclient["bytes"],
            "client_originated_byte_hop_reduction_vs_direct_pct": (
                (1 - hclient["bytes"] / dclient["bytes"]) * 100
                if dclient["bytes"] > 0 else None
            ),
            "s3to_edge_originated_byte_hops": hedge["bytes"],
            "direct_edge_originated_byte_hops": dedge["bytes"],
            "definition": (
                "Sum of IPv4 forwarding-rule byte counters across switch egress ports. "
                "This counts each traversed hop and is not unique traffic volume or "
                "application payload bytes."
            ),
        },
        "initial_model_sha256": hhash,
        "rounds": round_rows,
        "warnings": [
            "The comparison is valid only for the same host, topology, and load conditions.",
            "Direct-vs-hierarchical accuracy difference is measured, not assumed to be zero.",
            "Switch-observed byte sums are byte-hops and must not be reported as unique bytes.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchical", required=True, help="S3TO result JSON")
    parser.add_argument("--direct", required=True, help="Direct FedAvg result JSON")
    parser.add_argument("--output", required=True, help="Output comparison JSON")
    parser.add_argument("--accuracy-tolerance-pp", type=float, default=0.10)
    args = parser.parse_args()
    if not math.isfinite(args.accuracy_tolerance_pp) or args.accuracy_tolerance_pp < 0:
        parser.error("--accuracy-tolerance-pp must be a finite non-negative number")
    result = compare(args.hierarchical, args.direct, args.accuracy_tolerance_pp)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "saved_to": str(target),
        "final_accuracy_pct": result["final_accuracy_pct"],
        "global_host_link": result["global_host_link"],
        "network_load_proxy": result["network_load_proxy"],
        "wall_clock": result["wall_clock"],
    }, indent=2))


if __name__ == "__main__":
    main()
