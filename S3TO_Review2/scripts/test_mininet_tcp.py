#!/usr/bin/env python3
import argparse
import json
import socket
import threading
import time

from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.topo import Topo
from mininet.log import setLogLevel


class S3TOTestTopo(Topo):
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

        self.addLink(clients[0], s1, port2=1)
        self.addLink(clients[1], s1, port2=2)
        self.addLink(s1, s2, port1=3, port2=3)
        self.addLink(edge1, s1, port2=4)

        self.addLink(clients[2], s2, port2=1)
        self.addLink(clients[3], s2, port2=2)
        self.addLink(s2, s3, port1=4, port2=4)

        self.addLink(clients[4], s3, port2=1)
        self.addLink(clients[5], s3, port2=2)
        self.addLink(edge2, s3, port2=3)



def start_listener(host, port, results, lock, expected_connections=3):
    code = """
import socket, json
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', PORT))
s.listen(20)
print('LISTENER_READY', flush=True)
for _ in range(EXPECTED):
    c, a = s.accept()
    data = c.recv(65535)
    print('RECEIVED ' + data.decode(), flush=True)
    c.sendall(json.dumps({'ok': True}).encode())
    c.close()
s.close()
"""
    code = code.replace("PORT", str(port))
    code = code.replace("EXPECTED", str(expected_connections))
    proc = host.popen(["python3", "-u", "-c", code])

    deadline = time.time() + 10
    while time.time() < deadline:
        line = proc.stdout.readline()
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        if "LISTENER_READY" in line:
            return proc
        if proc.poll() is not None:
            raise RuntimeError(
                f"{host.name} listener exited early; output: {line}"
            )

    proc.terminate()
    raise TimeoutError(f"{host.name} listener did not become ready")

def run(args):
    topo = S3TOTestTopo()
    net = Mininet(
        topo=topo,
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

    procs = []
    try:
        net.start()
        edge1 = net.get("edge1")
        edge2 = net.get("edge2")

        # Receivers run inside the Mininet edge-host network namespaces.
        procs.append(start_listener(edge1, 9101, {}, threading.Lock()))
        procs.append(start_listener(edge2, 9102, {}, threading.Lock()))

        # Apply the known assignment through the Ryu UDP API.
        from network.telemetry import send_controller_command
        assignment = {
            0: "edge1", 1: "edge1", 2: "edge2",
            3: "edge1", 4: "edge2", 5: "edge2",
        }
        send_controller_command(
            args.controller_ip, 9999,
            {"type": "apply_assignment", "assignment": assignment},
        )
        time.sleep(1)

        routes = {
            0: ("edge1", 9101),
            1: ("edge1", 9101),
            2: ("edge2", 9102),
            3: ("edge1", 9101),
            4: ("edge2", 9102),
            5: ("edge2", 9102),
        }

        print("\n=== TCP MODEL-UPDATE PATH TEST ===")
        for cid, (edge_name, port) in routes.items():
            client = net.get(f"c{cid+1}")
            payload = json.dumps({
                "type": "test_model_update",
                "client_id": cid,
                "sample_count": 250,
            })
            edge_ip = "10.0.0.101" if edge_name == "edge1" else "10.0.0.102"
            client_code = """
import socket, sys
host = sys.argv[1]
port = int(sys.argv[2])
payload = sys.argv[3].encode()
with socket.create_connection((host, port), timeout=5) as s:
    s.sendall(payload)
    print(s.recv(1024).decode())
"""
            proc = client.popen(
                ["python3", "-c", client_code, edge_ip, str(port), payload],
                stdout=-1,
                stderr=2,
                text=True,
            )
            try:
                result, _ = proc.communicate(timeout=10)
            except Exception:
                proc.kill()
                proc.wait()
                raise

            print(f"Client {cid+1} -> {edge_name}: {result.strip()}")
            if proc.returncode != 0 or '"ok": true' not in result and '"ok":true' not in result:
                raise RuntimeError(f"TCP test failed for client {cid+1}: {result}")
            print(f"Client {cid+1} -> {edge_name}: {result.strip()}")
            if '"ok":true' not in result.replace(" ", ""):
                raise RuntimeError(f"TCP test failed for client {cid+1}: {result}")

        print("\nPASS: All six clients received TCP acknowledgements through Mininet.")
        print("This validates the network path, not yet full PyTorch aggregation.")
        print("\nInspect flow counters in Mininet CLI using:")
        print("  sh ovs-ofctl -O OpenFlow13 dump-flows s1")
        print("  sh ovs-ofctl -O OpenFlow13 dump-flows s2")
        print("  sh ovs-ofctl -O OpenFlow13 dump-flows s3")
    finally:
        for proc in procs:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                pass
        net.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-ip", default="127.0.0.1")
    parser.add_argument("--controller-port", type=int, default=6653)
    run_args = parser.parse_args()
    setLogLevel("warning")
    run(run_args)
