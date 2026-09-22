import json
import socket
import threading

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, ipv4

from s3to.engine import S3TOEngine


CLIENT_IPS = {
    0: "10.0.0.11",
    1: "10.0.0.12",
    2: "10.0.0.13",
    3: "10.0.0.14",
    4: "10.0.0.15",
    5: "10.0.0.16",
}
EDGE_IPS = {
    "edge1": "10.0.0.101",
    "edge2": "10.0.0.102",
}

# For the fixed Mininet topology in network/topology.py.
PATH_PORTS = {
    0: {
        "edge1": [(1, 4)],
        "edge2": [(1, 3), (2, 4), (3, 3)],
    },
    1: {
        "edge1": [(1, 4)],
        "edge2": [(1, 3), (2, 4), (3, 3)],
    },
    2: {
        "edge1": [(2, 3), (1, 4)],
        "edge2": [(2, 4), (3, 3)],
    },
    3: {
        "edge1": [(2, 3), (1, 4)],
        "edge2": [(2, 4), (3, 3)],
    },
    4: {
        "edge1": [(3, 4), (2, 3), (1, 4)],
        "edge2": [(3, 3)],
    },
    5: {
        "edge1": [(3, 4), (2, 3), (1, 4)],
        "edge2": [(3, 3)],
    },
}


class S3TORyu(app_manager.RyuApp):
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
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        self.logger.info("Switch connected: dpid=%s", datapath.id)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def state_change_handler(self, ev):
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
        sock.bind(("0.0.0.0", 9999))
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
        kind = msg.get("type", "telemetry")
        if kind == "telemetry":
            cid = int(msg["client_id"])
            self.telemetry[cid] = msg["histogram"]
            if len(self.telemetry) >= 6:
                result = self.engine.decide(self.telemetry)
                self.assignment = result.assignment
                self.logger.info("S3TO assignment: %s", self.assignment)
            sock.sendto(json.dumps({"status": "ready" if self.assignment else "collecting", "assignment": self.assignment}).encode(), addr)
        elif kind == "apply_assignment":
            self.assignment = {int(k): v for k, v in msg["assignment"].items()}
            self.install_assignment_flows()
            sock.sendto(json.dumps({"status": "flows_installed", "assignment": self.assignment}).encode(), addr)
        elif kind == "reset":
            self.telemetry.clear()
            self.assignment.clear()
            sock.sendto(json.dumps({"status": "reset"}).encode(), addr)

    def install_assignment_flows(self):
        parser_by_dpid = {dpid: dp.ofproto_parser for dpid, dp in self.datapaths.items()}
        for cid, edge in self.assignment.items():
            if cid not in PATH_PORTS:
                continue
            for dpid, out_port in PATH_PORTS[cid][edge]:
                datapath = self.datapaths.get(dpid)
                if not datapath:
                    continue
                parser = parser_by_dpid[dpid]
                match = parser.OFPMatch(
                    eth_type=ether_types.ETH_TYPE_IP,
                    ipv4_src=CLIENT_IPS[cid],
                    ipv4_dst=EDGE_IPS[edge],
                )
                actions = [parser.OFPActionOutput(out_port)]
                self.add_flow(datapath, 100, match, actions)
        self.logger.info("Installed S3TO OpenFlow rules")
