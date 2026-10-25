import json
import socket
import threading

from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import (
    CONFIG_DISPATCHER, DEAD_DISPATCHER, MAIN_DISPATCHER, set_ev_cls,
)
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib.packet import packet, ethernet, ether_types, ipv4

from s3to.engine import S3TOEngine
from network.mininet_paths import (
    CLIENT_IPS as DIRECT_CLIENT_IPS, GLOBAL_IP,
    DIRECT_UPLINK_PATHS, DIRECT_DOWNLINK_PATHS, ALL_EXPERIMENT_HOST_IPS,
    S3TO_UPLINK_PATHS, S3TO_DOWNLINK_PATHS,
)


CLIENT_IPS = DIRECT_CLIENT_IPS
EDGE_IPS = {
    "edge1": "10.0.0.101",
    "edge2": "10.0.0.102",
}

# Explicit routes for the fixed six-client S3TO topology.
PATH_PORTS = S3TO_UPLINK_PATHS

class S3TORyu(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = S3TOEngine(group_size=3)
        self.datapaths = {}
        self.mac_to_port = {}
        self.assignment = {}
        self.telemetry = {}
        self._udp_stop = threading.Event()
        self._udp_thread = threading.Thread(target=self._udp_server, daemon=True)
        self._udp_thread.start()

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        self.datapaths[datapath.id] = datapath
        self.mac_to_port[datapath.id] = {}
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.logger.info("Switch connected: dpid=%s", datapath.id)

    @set_ev_cls(
        ofp_event.EventOFPStateChange,
        [MAIN_DISPATCHER, CONFIG_DISPATCHER, DEAD_DISPATCHER],
    )
    def state_change_handler(self, ev):
        if ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(ev.datapath.id, None)
            self.mac_to_port.pop(ev.datapath.id, None)
            return
        self.datapaths[ev.datapath.id] = ev.datapath

    def add_flow(self, datapath, priority, match, actions, idle_timeout=0, hard_timeout=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
        )
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        dpid = datapath.id
        parser = datapath.ofproto_parser
        in_port = msg.match.get("in_port")
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if not eth:
            return
        self.mac_to_port.setdefault(dpid, {})[eth.src] = in_port
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        out_port = self.mac_to_port[dpid].get(eth.dst, datapath.ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=None if msg.buffer_id != datapath.ofproto.OFP_NO_BUFFER else msg.data,
        )
        datapath.send_msg(out)

    def _udp_server(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Control commands change switch forwarding. Keep this API loopback-only
        # so untrusted LAN hosts cannot install flows or reset the experiment.
        sock.bind(("127.0.0.1", 9999))
        sock.settimeout(0.5)
        while not self._udp_stop.is_set():
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data.decode("utf-8"))
                self._handle_udp(msg, addr, sock)
            except Exception as exc:
                self.logger.exception("UDP command error: %s", exc)
                sock.sendto(json.dumps({"status": "error", "message": str(exc)}).encode(), addr)

    def _handle_udp(self, msg, addr, sock):
        if not isinstance(msg, dict):
            raise ValueError("controller command must be a JSON object")
        kind = msg.get("type", "telemetry")
        if kind == "telemetry":
            try:
                cid = int(msg["client_id"])
                histogram = msg["histogram"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("telemetry requires client_id and histogram") from exc
            if cid not in CLIENT_IPS:
                raise ValueError("client_id must be in the range 0..5")
            if not isinstance(histogram, list) or len(histogram) != 10:
                raise ValueError("histogram must contain exactly 10 label counts")
            if any(type(count) is not int or count < 0 for count in histogram):
                raise ValueError("histogram counts must be non-negative integers")
            if sum(histogram) < 1:
                raise ValueError("histogram must contain at least one sample")
            supplied_ip = msg.get("client_ip")
            if supplied_ip and supplied_ip != CLIENT_IPS[cid]:
                raise ValueError("client_ip does not match client_id")
            self.telemetry[cid] = histogram
            if set(self.telemetry) == set(CLIENT_IPS):
                result = self.engine.decide(self.telemetry)
                self.assignment = result.assignment
                self.logger.info("S3TO assignment: %s", self.assignment)
                self.install_assignment_flows()
            sock.sendto(json.dumps({
                "status": "ready" if self.assignment else "collecting",
                "assignment": self.assignment,
                "received_clients": len(self.telemetry),
            }).encode(), addr)
        elif kind == "apply_assignment":
            raw_assignment = msg.get("assignment")
            if not isinstance(raw_assignment, dict):
                raise ValueError("assignment must be a mapping")
            try:
                assignment = {int(cid): edge for cid, edge in raw_assignment.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError("assignment client IDs must be integers") from exc
            if set(assignment) != set(CLIENT_IPS) or any(edge not in EDGE_IPS for edge in assignment.values()):
                raise ValueError("assignment must cover clients 0..5 exactly once with valid edges")
            counts = {edge: sum(value == edge for value in assignment.values()) for edge in EDGE_IPS}
            if any(count != 3 for count in counts.values()):
                raise ValueError("assignment must allocate exactly three clients to each edge")
            self.assignment = assignment
            self.install_assignment_flows()
            sock.sendto(json.dumps({"status": "flows_installed", "assignment": self.assignment}).encode(), addr)
        elif kind == "apply_direct":
            self.clear_assignment_flows()
            self.telemetry.clear()
            self.assignment.clear()
            self.install_direct_flows()
            sock.sendto(json.dumps({"status": "direct_flows_installed"}).encode(), addr)
        elif kind == "reset":
            self.clear_assignment_flows()
            self.telemetry.clear()
            self.assignment.clear()
            sock.sendto(json.dumps({"status": "reset"}).encode(), addr)
        else:
            sock.sendto(json.dumps({"status": "error", "message": "unknown command"}).encode(), addr)

    def clear_assignment_flows(self):
        # Remove all experiment IPv4 host rules on each switch. This must include
        # edge/global addresses too: otherwise edge->global rules survive a
        # hierarchical run and contaminate the following direct baseline counters.
        for datapath in list(self.datapaths.values()):
            parser = datapath.ofproto_parser
            ofproto = datapath.ofproto
            for client_ip in ALL_EXPERIMENT_HOST_IPS:
                for direction in ("ipv4_src", "ipv4_dst"):
                    match_fields = {"eth_type": ether_types.ETH_TYPE_IP, direction: client_ip}
                    match = parser.OFPMatch(**match_fields)
                    mod = parser.OFPFlowMod(
                        datapath=datapath,
                        command=ofproto.OFPFC_DELETE,
                        table_id=ofproto.OFPTT_ALL,
                        out_port=ofproto.OFPP_ANY,
                        out_group=ofproto.OFPG_ANY,
                        match=match,
                    )
                    datapath.send_msg(mod)
        self.logger.info("Cleared previous client forwarding rules")

    def install_direct_flows(self):
        """Install explicit bidirectional routes for direct FedAvg traffic."""
        for cid, client_ip in CLIENT_IPS.items():
            for dpid, out_port in DIRECT_UPLINK_PATHS[cid]:
                datapath = self.datapaths.get(dpid)
                if datapath is None:
                    raise RuntimeError(f"OpenFlow switch dpid={dpid} is not connected")
                parser = datapath.ofproto_parser
                match = parser.OFPMatch(
                    eth_type=ether_types.ETH_TYPE_IP,
                    ipv4_src=client_ip,
                    ipv4_dst=GLOBAL_IP,
                )
                self.add_flow(datapath, 110, match, [parser.OFPActionOutput(out_port)])

            for dpid, out_port in DIRECT_DOWNLINK_PATHS[cid]:
                datapath = self.datapaths.get(dpid)
                if datapath is None:
                    raise RuntimeError(f"OpenFlow switch dpid={dpid} is not connected")
                parser = datapath.ofproto_parser
                match = parser.OFPMatch(
                    eth_type=ether_types.ETH_TYPE_IP,
                    ipv4_src=GLOBAL_IP,
                    ipv4_dst=client_ip,
                )
                self.add_flow(datapath, 110, match, [parser.OFPActionOutput(out_port)])

        self.logger.info("Installed direct client-to-global FedAvg OpenFlow rules")

    def install_assignment_flows(self):
        parser_by_dpid = {dpid: dp.ofproto_parser for dpid, dp in self.datapaths.items()}
        for cid, edge in self.assignment.items():
            if cid not in PATH_PORTS:
                continue
            for dpid, out_port in PATH_PORTS[cid][edge]:
                datapath = self.datapaths.get(dpid)
                if datapath is None:
                    raise RuntimeError(f"OpenFlow switch dpid={dpid} is not connected")
                parser = parser_by_dpid[dpid]
                match = parser.OFPMatch(
                    eth_type=ether_types.ETH_TYPE_IP,
                    ipv4_src=CLIENT_IPS[cid],
                    ipv4_dst=EDGE_IPS[edge],
                )
                actions = [parser.OFPActionOutput(out_port)]
                self.add_flow(datapath, 100, match, actions)

            # Install reverse edge-to-client rules as well. Without these, replies
            # fall through to table-miss flooding and the measured byte-hop proxy
            # undercounts S3TO response traffic.
            for dpid, out_port in S3TO_DOWNLINK_PATHS[cid][edge]:
                datapath = self.datapaths.get(dpid)
                if datapath is None:
                    raise RuntimeError(f"OpenFlow switch dpid={dpid} is not connected")
                parser = datapath.ofproto_parser
                match = parser.OFPMatch(
                    eth_type=ether_types.ETH_TYPE_IP,
                    ipv4_src=EDGE_IPS[edge],
                    ipv4_dst=CLIENT_IPS[cid],
                )
                self.add_flow(
                    datapath, 100, match, [parser.OFPActionOutput(out_port)]
                )

        # Explicit paths for edge-to-global aggregation and TCP responses.
        aggregator_paths = (
            ("10.0.0.101", "10.0.0.103", ((1, 3), (2, 5))),
            ("10.0.0.103", "10.0.0.101", ((2, 3), (1, 4))),
            ("10.0.0.102", "10.0.0.103", ((3, 4), (2, 5))),
            ("10.0.0.103", "10.0.0.102", ((2, 4), (3, 3))),
        )
        for src_ip, dst_ip, path in aggregator_paths:
            for dpid, out_port in path:
                datapath = self.datapaths.get(dpid)
                if datapath is None:
                    raise RuntimeError(f"OpenFlow switch dpid={dpid} is not connected")
                parser = datapath.ofproto_parser
                match = parser.OFPMatch(
                    eth_type=ether_types.ETH_TYPE_IP,
                    ipv4_src=src_ip,
                    ipv4_dst=dst_ip,
                )
                self.add_flow(
                    datapath, 100, match,
                    [parser.OFPActionOutput(out_port)],
                )

        self.logger.info("Installed S3TO OpenFlow rules")
