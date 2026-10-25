"""Parse OpenFlow counters without double-counting the global-aggregator link."""
import re

_NUMBER = r"(\d+)"


def parse_openflow_flows(raw):
    """Return parsed OVS flow-counter rows from `ovs-ofctl dump-flows` text."""
    rows = []
    for line in raw.splitlines():
        packets = re.search(r"\bn_packets=" + _NUMBER, line)
        bytes_match = re.search(r"\bn_bytes=" + _NUMBER, line)
        if not packets or not bytes_match:
            continue
        source = re.search(r"\b(?:nw_src|ipv4_src)=([0-9.]+)", line)
        destination = re.search(r"\b(?:nw_dst|ipv4_dst)=([0-9.]+)", line)
        input_port = re.search(r"\bin_port=" + _NUMBER, line)
        output_port = re.search(r"\boutput:" + _NUMBER, line)
        rows.append({
            "packets": int(packets.group(1)),
            "bytes": int(bytes_match.group(1)),
            "src_ip": source.group(1) if source else None,
            "dst_ip": destination.group(1) if destination else None,
            "in_port": int(input_port.group(1)) if input_port else None,
            "out_port": int(output_port.group(1)) if output_port else None,
            "raw": line.strip(),
        })
    return rows


def summarize_pathA_metrics(per_switch_flows, aggregation_mode):
    """Summarize s2's unique global-host link counters and switch byte-hops.

    OVS flow counters are per-switch counters, so summing all switches is not a
    unique-byte total. The specific global link is represented at s2 by output
    port 5 for traffic sent to the global host (10.0.0.103), and source address
    10.0.0.103 for traffic returning from it.
    """
    if aggregation_mode not in {"hierarchical", "direct"}:
        raise ValueError("aggregation_mode must be hierarchical or direct")
    global_ip = "10.0.0.103"
    edge_ips = {"10.0.0.101", "10.0.0.102"}
    client_ips = {f"10.0.0.{11 + cid}" for cid in range(6)}
    switch_byte_sum = sum(
        row["bytes"]
        for rows in per_switch_flows.values()
        for row in rows
    )
    switch_packet_sum = sum(
        row["packets"]
        for rows in per_switch_flows.values()
        for row in rows
    )
    # Count only IPv4 forwarding rules with a concrete physical output port.
    # Each packet is counted once per traversed switch hop, so this is a
    # byte-hop/load metric, not a unique-byte metric. It avoids counting
    # table-miss packet-ins or OpenFlow controller output as link traffic.
    physical_ports = {
        "s1": {1, 2, 3, 4},
        "s2": {1, 2, 3, 4, 5},
        "s3": {1, 2, 3, 4},
    }
    egress_rows = [
        row
        for switch_name, rows in per_switch_flows.items()
        for row in rows
        if row["src_ip"] is not None
        and row["dst_ip"] is not None
        and row["out_port"] in physical_ports.get(switch_name, set())
        and (row["packets"] > 0 or row["bytes"] > 0)
    ]
    central = per_switch_flows.get("s2", [])
    uploads = [
        row for row in central
        if row["dst_ip"] == global_ip and row["out_port"] == 5
        and (row["packets"] > 0 or row["bytes"] > 0)
    ]
    responses_from_global = [
        row for row in central
        if row["src_ip"] == global_ip and row["out_port"] in {1, 2, 3, 4}
        and (row["packets"] > 0 or row["bytes"] > 0)
    ]

    def aggregate(rows):
        return {
            "packets": sum(row["packets"] for row in rows),
            "bytes": sum(row["bytes"] for row in rows),
            "flow_count": len(rows),
        }

    edge_uploads = [row for row in uploads if row["src_ip"] in edge_ips]
    direct_uploads = [row for row in uploads if row["src_ip"] in client_ips]
    edge_responses = [row for row in responses_from_global if row["dst_ip"] in edge_ips]
    client_responses = [row for row in responses_from_global if row["dst_ip"] in client_ips]
    unexpected_responses = [
        row for row in responses_from_global
        if row["dst_ip"] not in edge_ips and row["dst_ip"] not in client_ips
    ]
    unexpected_uploads = [
        row for row in uploads
        if row["src_ip"] not in edge_ips and row["src_ip"] not in client_ips
    ]
    if not uploads:
        raise ValueError("No measured uploads crossed s2 port 5 to the global host")
    if aggregation_mode == "direct" and edge_uploads:
        raise ValueError("Direct baseline unexpectedly has edge-to-global flows")
    if aggregation_mode == "direct" and not direct_uploads:
        raise ValueError("Direct baseline has no client-to-global upload counters")
    if aggregation_mode == "hierarchical" and direct_uploads:
        raise ValueError("Hierarchical run unexpectedly has client-to-global flows")
    if aggregation_mode == "hierarchical" and not edge_uploads:
        raise ValueError("Hierarchical run has no edge-to-global upload counters")

    client_rows = [row for row in egress_rows if row["src_ip"] in client_ips]
    edge_rows = [row for row in egress_rows if row["src_ip"] in edge_ips]
    global_rows = [row for row in egress_rows if row["src_ip"] == global_ip]
    known_sources = client_ips | edge_ips | {global_ip}
    other_rows = [row for row in egress_rows if row["src_ip"] not in known_sources]

    return {
        "aggregation_mode": aggregation_mode,
        "ipv4_forwarding_egress_byte_hops": {
            "all_origins": aggregate(egress_rows),
            "client_originated": aggregate(client_rows),
            "edge_originated": aggregate(edge_rows),
            "global_originated": aggregate(global_rows),
            "other_originated": aggregate(other_rows),
            "definition": (
                "Sum of IPv4 forwarding-rule counters with physical output ports. "
                "Each traversal across a switch egress contributes bytes, so transit "
                "traffic is counted once per hop; this is a network-load proxy, not "
                "unique end-to-end bytes or application payload bytes."
            ),
        },
        "global_host_link_at_s2_port5": {
            "uploads_to_global": aggregate(uploads),
            "responses_from_global": aggregate(responses_from_global),
            "edge_to_global_uploads": aggregate(edge_uploads),
            "client_to_global_uploads": aggregate(direct_uploads),
            "edge_responses_from_global": aggregate(edge_responses),
            "client_responses_from_global": aggregate(client_responses),
            "unexpected_uploads": aggregate(unexpected_uploads),
            "unexpected_responses_from_global": aggregate(unexpected_responses),
            "bidirectional_bytes": aggregate(uploads)["bytes"] + aggregate(responses_from_global)["bytes"],
            "scope_note": (
                "These are OpenFlow forwarding counters for model-update transport and "
                "its TCP/application responses. The current runner does not transmit the "
                "global model down to clients over Mininet, so these are not a complete "
                "accounting of all possible federated-learning communication."
            ),
        },
        "raw_switch_flow_counter_sums_including_non_forwarding_rules": {
            "packet_sum": switch_packet_sum,
            "byte_sum": switch_byte_sum,
            "warning": (
                "These raw sums can include non-forwarding flow entries and duplicate "
                "transit traffic; do not report them as unique bytes or payload bytes. "
                "Use ipv4_forwarding_egress_byte_hops for the forwarding-load proxy."
            ),
        },
    }
