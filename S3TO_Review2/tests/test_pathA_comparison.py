import json
import os
import socket
import struct
import subprocess
import sys
import time
import unittest
from pathlib import Path

import torch

from fl.fedavg import fedavg
from network.openflow_metrics import parse_openflow_flows, summarize_pathA_metrics
from network.secure_serialization import (
    MAX_MESSAGE_BYTES, pack_message, recv_message,
)
from network.secure_tcp import send_tcp_message, ThreadedTCPServer
from network.mininet_paths import (
    CLIENT_IPS, DIRECT_UPLINK_PATHS, DIRECT_DOWNLINK_PATHS,
    ALL_EXPERIMENT_HOST_IPS, S3TO_UPLINK_PATHS, S3TO_DOWNLINK_PATHS,
    validate_direct_paths, validate_s3to_paths,
)


class TestSafeMininetSerialization(unittest.TestCase):
    def round_trip(self, value):
        left, right = socket.socketpair()
        try:
            left.sendall(pack_message(value))
            return recv_message(right)
        finally:
            left.close()
            right.close()

    def test_nested_tensor_round_trip_preserves_dtype_and_shape(self):
        payload = {
            "type": "model_update",
            "sample_count": 500,
            "state_dict": {
                "weights": torch.tensor([[1.25, -2.0], [3.5, 4.0]], dtype=torch.float32),
                "counter": torch.tensor(7, dtype=torch.int64),
                "mask": torch.tensor([True, False], dtype=torch.bool),
                "empty": torch.empty((0, 2), dtype=torch.float16),
            },
        }
        result = self.round_trip(payload)
        self.assertEqual(result["sample_count"], 500)
        for key, value in payload["state_dict"].items():
            torch.testing.assert_close(result["state_dict"][key], value)
            self.assertEqual(result["state_dict"][key].dtype, value.dtype)
            self.assertEqual(result["state_dict"][key].shape, value.shape)

    def test_reserved_marker_is_preserved_as_an_ordinary_mapping_key(self):
        value = {"__s3to_tensor_v1__": {"user_data": "not a tensor"}}
        self.assertEqual(self.round_trip(value), value)

    def test_unsupported_python_objects_are_rejected(self):
        with self.assertRaises(TypeError):
            pack_message({"bad": object()})

    def test_nonfinite_float_is_rejected(self):
        with self.assertRaises(ValueError):
            pack_message({"value": float("nan")})

    def test_oversized_frame_is_rejected_before_payload_read(self):
        left, right = socket.socketpair()
        try:
            left.sendall(struct.pack("!I", MAX_MESSAGE_BYTES + 1))
            with self.assertRaises(ValueError):
                recv_message(right)
        finally:
            left.close()
            right.close()

    def test_nonfinite_json_value_from_peer_is_rejected(self):
        left, right = socket.socketpair()
        try:
            raw = b'{"__s3to_dict_v1__":[["value",NaN]]}'
            left.sendall(struct.pack("!I", len(raw)) + raw)
            with self.assertRaises(ValueError):
                recv_message(right)
        finally:
            left.close()
            right.close()

    def test_duplicate_raw_json_object_keys_are_rejected(self):
        left, right = socket.socketpair()
        try:
            raw = b'{"a":1,"a":2}'
            left.sendall(struct.pack("!I", len(raw)) + raw)
            with self.assertRaisesRegex(ValueError, "Duplicate JSON object key"):
                recv_message(right)
        finally:
            left.close()
            right.close()

    def test_duplicate_json_keys_are_rejected(self):
        left, right = socket.socketpair()
        try:
            raw = b'{"__s3to_dict_v1__":[["sample_count",1],["sample_count",999]]}'
            left.sendall(struct.pack("!I", len(raw)) + raw)
            with self.assertRaisesRegex(ValueError, "Duplicate mapping key"):
                recv_message(right)
        finally:
            left.close()
            right.close()

    def test_pickle_payload_is_not_deserialized(self):
        import pickle
        left, right = socket.socketpair()
        try:
            payload = pickle.dumps({"type": "ping"})
            left.sendall(struct.pack("!I", len(payload)) + payload)
            with self.assertRaises(ValueError):
                recv_message(right)
        finally:
            left.close()
            right.close()

    def test_standalone_pi_wire_codec_matches_server_codec(self):
        from scripts.secure_wire import pack_message as pi_pack
        from scripts.secure_wire import recv_message as pi_recv

        payload = {
            "type": "model_update",
            "client_id": "pi1",
            "sample_count": 2,
            "state_dict": {
                "weight": torch.tensor([[1.5, -2.0]], dtype=torch.float32),
                "counter": torch.tensor(3, dtype=torch.int64),
            },
        }
        left, right = socket.socketpair()
        try:
            left.sendall(pi_pack(payload))
            decoded_by_server_codec = recv_message(right)
            self.assertEqual(decoded_by_server_codec["client_id"], "pi1")
            torch.testing.assert_close(
                decoded_by_server_codec["state_dict"]["weight"],
                payload["state_dict"]["weight"],
            )
        finally:
            left.close()
            right.close()

        left, right = socket.socketpair()
        try:
            left.sendall(pack_message(payload))
            decoded_by_pi_codec = pi_recv(right)
            self.assertEqual(decoded_by_pi_codec["client_id"], "pi1")
            torch.testing.assert_close(
                decoded_by_pi_codec["state_dict"]["weight"],
                payload["state_dict"]["weight"],
            )
        finally:
            left.close()
            right.close()

    def test_direct_global_service_rejects_client_id_spoofed_from_localhost(self):
        root = Path(__file__).resolve().parents[1]
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        proc = subprocess.Popen(
            [
                sys.executable, "scripts/mininet_aggregator_service.py", "global",
                "--port", str(port), "--aggregation-mode", "direct",
                "--expected-clients", "2",
            ],
            cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env={**os.environ, "PYTHONPATH": str(root)},
        )
        try:
            ready_deadline = time.monotonic() + 10
            while time.monotonic() < ready_deadline:
                line = proc.stdout.readline()
                if "SERVICE_READY" in line:
                    break
                if proc.poll() is not None:
                    self.fail("Direct aggregation service exited before ready")
            else:
                self.fail("Direct aggregation service did not become ready")
            response = send_tcp_message("127.0.0.1", port, {
                "type": "model_update", "round": 1, "client_id": 0,
                "sample_count": 1, "state_dict": {"w": torch.tensor([1.0])},
            })
            self.assertFalse(response["ok"])
            self.assertIn("peer address", response["error"])
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)

    def test_secure_tcp_server_round_trip(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        server = ThreadedTCPServer(
            "127.0.0.1", port,
            lambda msg, addr: {"ok": True, "echo": msg["payload"]},
        )
        server.start()
        try:
            response = send_tcp_message(
                "127.0.0.1", port,
                {"payload": torch.tensor([1.0, 2.0])}, timeout=3,
            )
            self.assertTrue(response["ok"])
            torch.testing.assert_close(response["echo"], torch.tensor([1.0, 2.0]))
        finally:
            server.stop()


    def test_direct_global_service_aggregates_client_updates_and_rejects_duplicates(self):
        root = Path(__file__).resolve().parents[1]
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        proc = subprocess.Popen(
            [
                sys.executable, "scripts/mininet_aggregator_service.py", "global",
                "--port", str(port), "--aggregation-mode", "direct",
                "--expected-clients", "2", "--allow-nonmininet-peers",
            ],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONPATH": str(root)},
        )
        try:
            ready_deadline = time.monotonic() + 10
            ready_line = ""
            while time.monotonic() < ready_deadline:
                ready_line = proc.stdout.readline()
                if "SERVICE_READY" in ready_line:
                    break
                if proc.poll() is not None:
                    self.fail("Direct aggregation service exited before ready")
            self.assertIn("SERVICE_READY global", ready_line)
            self.assertTrue(send_tcp_message("127.0.0.1", port, {
                "type": "reset_round", "round": 1,
            })["ok"])
            first = send_tcp_message("127.0.0.1", port, {
                "type": "model_update", "round": 1, "client_id": 0,
                "sample_count": 1, "state_dict": {"w": torch.tensor([1.0])},
            })
            second = send_tcp_message("127.0.0.1", port, {
                "type": "model_update", "round": 1, "client_id": 1,
                "sample_count": 3, "state_dict": {"w": torch.tensor([3.0])},
            })
            self.assertEqual(first["received_clients"], 1)
            self.assertEqual(second["received_clients"], 2)
            duplicate = send_tcp_message("127.0.0.1", port, {
                "type": "model_update", "round": 1, "client_id": 1,
                "sample_count": 3, "state_dict": {"w": torch.tensor([9.0])},
            })
            self.assertFalse(duplicate["ok"])
            result = send_tcp_message("127.0.0.1", port, {
                "type": "get_global", "round": 1,
            })
            self.assertTrue(result["ok"])
            self.assertEqual(result["sample_count"], 4)
            self.assertEqual(result["updates_aggregated"], 2)
            torch.testing.assert_close(result["state_dict"]["w"], torch.tensor([2.5]))
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)



class TestFedAvgValidationAndEquivalence(unittest.TestCase):
    def test_hierarchical_weighted_average_matches_direct_weighted_average(self):
        client_states = [
            {"w": torch.tensor([float(i), float(2 * i)]), "n": torch.tensor(i)}
            for i in range(1, 7)
        ]
        counts = [500] * 6
        direct = fedavg(client_states, counts)
        edge1 = fedavg(client_states[:3], counts[:3])
        edge2 = fedavg(client_states[3:], counts[3:])
        hierarchical = fedavg([edge1, edge2], [sum(counts[:3]), sum(counts[3:])])
        torch.testing.assert_close(hierarchical["w"], direct["w"], rtol=1e-6, atol=1e-6)
        self.assertEqual(hierarchical["n"].item(), direct["n"].item())

    def test_fedavg_rejects_nonpositive_sample_count(self):
        with self.assertRaises(ValueError):
            fedavg([{"w": torch.tensor([1.0])}], [0])

    def test_fedavg_rejects_mismatched_keys(self):
        with self.assertRaises(ValueError):
            fedavg(
                [{"w": torch.tensor([1.0])}, {"other": torch.tensor([2.0])}],
                [1, 1],
            )

    def test_fedavg_rejects_mismatched_tensor_shape(self):
        with self.assertRaises(ValueError):
            fedavg(
                [{"w": torch.tensor([1.0])}, {"w": torch.tensor([2.0, 3.0])}],
                [1, 1],
            )


class TestModelFingerprint(unittest.TestCase):
    def test_fingerprint_is_deterministic_and_detects_weight_changes(self):
        from common.model_fingerprint import state_dict_sha256

        original = {"weight": torch.tensor([1.0, 2.0]), "counter": torch.tensor(3)}
        same_values_different_order = {"counter": torch.tensor(3), "weight": torch.tensor([1.0, 2.0])}
        self.assertEqual(
            state_dict_sha256(original),
            state_dict_sha256(same_values_different_order),
        )
        changed = {"weight": torch.tensor([1.0, 2.1]), "counter": torch.tensor(3)}
        self.assertNotEqual(state_dict_sha256(original), state_dict_sha256(changed))

class TestIntegerTelemetryHistogram(unittest.TestCase):
    def test_controller_histogram_conversion_uses_actual_integer_counts(self):
        from fl.data import TensorDataset, label_histogram

        dataset = TensorDataset(
            torch.zeros((4, 1, 28, 28)),
            torch.tensor([0, 0, 1, 1]),
        )
        encoded = [int(value) for value in label_histogram(dataset, [0, 1, 2, 3]).tolist()]
        self.assertEqual(encoded[:2], [2, 2])
        self.assertTrue(all(type(value) is int for value in encoded))

class TestPartitionFingerprint(unittest.TestCase):
    def test_partition_fingerprints_detect_changed_samples_and_preserve_order(self):
        from torch.utils.data import TensorDataset
        from common.data_fingerprint import partition_sha256_by_client

        dataset = TensorDataset(
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            torch.tensor([0, 1, 0]),
        )
        partitions = {0: [0, 2], 1: [1]}
        first = partition_sha256_by_client(dataset, partitions)
        self.assertEqual(first, partition_sha256_by_client(dataset, partitions))
        reordered = {0: [2, 0], 1: [1]}
        self.assertNotEqual(first, partition_sha256_by_client(dataset, reordered))

class TestDirectOpenFlowPaths(unittest.TestCase):
    def test_paths_cover_all_clients_and_use_global_link(self):
        self.assertTrue(validate_direct_paths())
        self.assertEqual(set(CLIENT_IPS), set(range(6)))
        self.assertEqual(set(DIRECT_UPLINK_PATHS), set(range(6)))
        self.assertEqual(set(DIRECT_DOWNLINK_PATHS), set(range(6)))
        self.assertEqual(
            set(ALL_EXPERIMENT_HOST_IPS),
            set(CLIENT_IPS.values()) | {"10.0.0.101", "10.0.0.102", "10.0.0.103"},
        )
        for cid in range(6):
            self.assertEqual(DIRECT_UPLINK_PATHS[cid][-1], (2, 5))
        self.assertEqual(DIRECT_UPLINK_PATHS[0], ((1, 3), (2, 5)))
        self.assertEqual(DIRECT_UPLINK_PATHS[2], ((2, 5),))
        self.assertEqual(DIRECT_UPLINK_PATHS[4], ((3, 4), (2, 5)))

    def test_s3to_paths_cover_each_client_and_both_edges_both_directions(self):
        self.assertTrue(validate_s3to_paths())
        self.assertEqual(set(S3TO_UPLINK_PATHS), set(range(6)))
        self.assertEqual(set(S3TO_DOWNLINK_PATHS), set(range(6)))
        for cid in range(6):
            self.assertEqual(set(S3TO_UPLINK_PATHS[cid]), {"edge1", "edge2"})
            self.assertEqual(set(S3TO_DOWNLINK_PATHS[cid]), {"edge1", "edge2"})
            for edge in ("edge1", "edge2"):
                self.assertTrue(S3TO_UPLINK_PATHS[cid][edge])
                self.assertTrue(S3TO_DOWNLINK_PATHS[cid][edge])

    def test_path_validator_rejects_a_missing_client(self):
        from network import mininet_paths
        original = mininet_paths.DIRECT_UPLINK_PATHS.pop(5)
        try:
            with self.assertRaises(ValueError):
                mininet_paths.validate_direct_paths()
        finally:
            mininet_paths.DIRECT_UPLINK_PATHS[5] = original



class TestOpenFlowMetrics(unittest.TestCase):
    def test_parser_and_direct_global_link_metrics(self):
        raw_s2 = """
 cookie=0x0, duration=4.0s, table=0, n_packets=12, n_bytes=1200, priority=110,ip,nw_src=10.0.0.11,nw_dst=10.0.0.103 actions=output:5
 cookie=0x0, duration=4.0s, table=0, n_packets=10, n_bytes=1000, priority=110,ip,nw_src=10.0.0.12,nw_dst=10.0.0.103 actions=output:5
 cookie=0x0, duration=4.0s, table=0, n_packets=6, n_bytes=600, priority=110,ip,nw_src=10.0.0.103,nw_dst=10.0.0.11 actions=output:3
"""
        rows = parse_openflow_flows(raw_s2)
        self.assertEqual(len(rows), 3)
        metrics = summarize_pathA_metrics({"s2": rows}, "direct")
        link = metrics["global_host_link_at_s2_port5"]
        self.assertEqual(link["uploads_to_global"]["packets"], 22)
        self.assertEqual(link["uploads_to_global"]["bytes"], 2200)
        self.assertEqual(link["client_to_global_uploads"]["flow_count"], 2)
        self.assertEqual(link["responses_from_global"]["bytes"], 600)
        egress = metrics["ipv4_forwarding_egress_byte_hops"]["all_origins"]
        self.assertEqual(egress["bytes"], 2800)

    def test_hierarchical_metrics_classify_edge_uploads(self):
        raw_s2 = """
 cookie=0x0, n_packets=5, n_bytes=500, priority=100,ip,nw_src=10.0.0.101,nw_dst=10.0.0.103 actions=output:5
 cookie=0x0, n_packets=7, n_bytes=700, priority=100,ip,nw_src=10.0.0.102,nw_dst=10.0.0.103 actions=output:5
"""
        metrics = summarize_pathA_metrics({"s2": parse_openflow_flows(raw_s2)}, "hierarchical")
        link = metrics["global_host_link_at_s2_port5"]
        self.assertEqual(link["edge_to_global_uploads"]["packets"], 12)
        self.assertEqual(link["client_to_global_uploads"]["packets"], 0)

    def test_zero_counter_stale_edge_rules_do_not_fail_direct_metrics(self):
        raw_s2 = """
 cookie=0x0, n_packets=0, n_bytes=0, priority=100,ip,nw_src=10.0.0.101,nw_dst=10.0.0.103 actions=output:5
 cookie=0x0, n_packets=3, n_bytes=300, priority=110,ip,nw_src=10.0.0.11,nw_dst=10.0.0.103 actions=output:5
"""
        metrics = summarize_pathA_metrics(
            {"s2": parse_openflow_flows(raw_s2)}, "direct"
        )
        link = metrics["global_host_link_at_s2_port5"]
        self.assertEqual(link["uploads_to_global"]["packets"], 3)
        self.assertEqual(link["edge_to_global_uploads"]["flow_count"], 0)
        self.assertEqual(
            metrics["ipv4_forwarding_egress_byte_hops"]["all_origins"]["flow_count"],
            1,
        )


class TestPathAComparisonReport(unittest.TestCase):
    def test_comparison_rejects_unmatched_config_and_reports_global_link(self):
        from scripts.compare_pathA_results import compare

        config = {
            "seed": 42, "partition_mode": "fixed", "dirichlet_alpha": 0.5,
            "num_clients": 6, "samples_per_client": 500, "local_epochs": 5,
            "batch_size": 64, "learning_rate": 0.01, "rounds": 1, "device": "cpu",
        }
        histograms = {str(i): [500, 0] for i in range(6)}
        def result(mode, accuracy, upload_bytes, source_kind):
            upload = {"packets": 10, "bytes": upload_bytes, "flow_count": 2}
            return {
                "aggregation_mode": mode, "dataset": "real_mnist", "model": "mlp",
                "config": config, "histograms": histograms,
                "initial_model_sha256": "a" * 64,
                "partition_sha256_by_client": {str(i): str(i) * 64 for i in range(6)},
                "data_summary": {
                    "full_training_examples_available": 60000,
                    "test_examples": 10000,
                    "federated_training_examples_per_round": 3000,
                },
                "results": [{"round": 1, "accuracy": accuracy, "wall_clock_sec": 10.0}],
                "network_metrics": {
                    "global_host_link_at_s2_port5": {
                        "uploads_to_global": upload,
                        "responses_from_global": {"packets": 1, "bytes": 10, "flow_count": 1},
                    },
                    "ipv4_forwarding_egress_byte_hops": {
                        "all_origins": {"packets": 10, "bytes": upload_bytes * 2, "flow_count": 2},
                        "client_originated": {"packets": 10, "bytes": upload_bytes, "flow_count": 2},
                        "edge_originated": {"packets": 0, "bytes": 0, "flow_count": 0},
                    },
                },
            }
        with __import__("tempfile").TemporaryDirectory() as temp:
            from pathlib import Path
            import json
            root = Path(temp)
            hpath, dpath = root / "h.json", root / "d.json"
            hpath.write_text(json.dumps(result("hierarchical", 0.4151, 1000, "edge")))
            dpath.write_text(json.dumps(result("direct", 0.4151, 6000, "client")))
            report = compare(hpath, dpath)
            self.assertEqual(report["global_host_link"]["s3to_upload_bytes"], 1000)
            self.assertAlmostEqual(
                report["global_host_link"]["s3to_upload_byte_reduction_vs_direct_pct"],
                83.3333333333, places=5,
            )
            self.assertEqual(report["network_load_proxy"]["s3to_ipv4_forwarding_byte_hops"], 2000)
            bad_config = result("direct", 0.4151, 6000, "client")
            bad_config["config"] = dict(config, learning_rate=0.02)
            dpath.write_text(json.dumps(bad_config))
            with self.assertRaises(ValueError):
                compare(hpath, dpath)
            bad_config = result("direct", 0.4151, 6000, "client")
            bad_config["initial_model_sha256"] = "b" * 64
            dpath.write_text(json.dumps(bad_config))
            with self.assertRaisesRegex(ValueError, "Initial model weights differ"):
                compare(hpath, dpath)
            bad_config = result("direct", 0.4151, 6000, "client")
            bad_config["partition_sha256_by_client"] = dict(bad_config["partition_sha256_by_client"], **{"0": "f" * 64})
            dpath.write_text(json.dumps(bad_config))
            with self.assertRaisesRegex(ValueError, "training examples differ"):
                compare(hpath, dpath)


if __name__ == "__main__":
    unittest.main()
