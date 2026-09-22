from mininet.topo import Topo


class S3TOTopo(Topo):
    """Fixed Review-2 emulation topology with deterministic port numbering."""

    def build(self):
        s1 = self.addSwitch("s1", dpid="0000000000000001")
        s2 = self.addSwitch("s2", dpid="0000000000000002")
        s3 = self.addSwitch("s3", dpid="0000000000000003")

        clients = []
        for i in range(1, 7):
            h = self.addHost(f"c{i}", ip=f"10.0.0.{10+i}/24")
            clients.append(h)
        edge1 = self.addHost("edge1", ip="10.0.0.101/24")
        edge2 = self.addHost("edge2", ip="10.0.0.102/24")

        # Deterministic switch ports.
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
