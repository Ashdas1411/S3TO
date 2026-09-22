#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

from network.telemetry import send_controller_command


def run(args):
    try:
        from mininet.net import Mininet
        from mininet.node import RemoteController, OVSKernelSwitch
        from mininet.cli import CLI
        from mininet.log import setLogLevel, info
        from network.topology import S3TOTopo
    except ImportError as exc:
        print("Mininet is not installed. Run this inside WSL/Ubuntu after installing Mininet.")
        print(exc)
        sys.exit(1)

    setLogLevel("info")
    topo = S3TOTopo()
    net = Mininet(
        topo=topo,
        controller=None,
        switch=OVSKernelSwitch,
        autoSetMacs=True,
        autoStaticArp=True,
    )
    controller = net.addController(
        "c0",
        controller=RemoteController,
        ip=args.controller_ip,
        port=args.controller_port,
    )

    net.start()
    try:
        info("*** Testing baseline connectivity\n")
        net.pingAll()
        info("*** Hosts:\n")
        for h in net.hosts:
            info(f"{h.name}: {h.IP()} / MAC {h.MAC()}\n")

        assignment = {0: "edge2", 1: "edge2", 2: "edge1", 3: "edge1", 4: "edge1", 5: "edge2"}
        info("*** Sending demo S3TO assignment to Ryu\n")
        send_controller_command(
            args.controller_ip,
            9999,
            {"type": "apply_assignment", "assignment": assignment},
        )
        time.sleep(1)
        info("*** Assignment sent. Inspect flows with: ovs-ofctl -O OpenFlow13 dump-flows s1\n")

        if args.cli:
            CLI(net)
    finally:
        net.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--controller-ip", default="127.0.0.1")
    p.add_argument("--controller-port", type=int, default=6653)
    p.add_argument("--cli", action="store_true")
    args = p.parse_args()
    if os.geteuid() != 0:
        print("Run Mininet as root: sudo python3 run_mininet_demo.py --cli")
        sys.exit(1)
    run(args)
