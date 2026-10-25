# S3TO hardware demonstration: scope and claims

## Software SDN experiment
The six-client Mininet experiment runs through Ryu-controlled OpenFlow switches.
Client model updates are routed to two Mininet edge aggregators; the edge
updates are then sent to the global aggregator. OpenFlow counters are collected
from the emulated switches.

## Physical Raspberry Pi experiment
The initial hardware demonstration uses two physical Raspberry Pi clients
connected to the laptop over the phone hotspot. Each client can test network
reachability and, when PyTorch is available, train a local MLP on its exported
MNIST partition and send its model update to the laptop receiver.

The hotspot is an ordinary Wi-Fi network, not the Mininet OpenFlow topology.
Therefore, physical client traffic is not claimed to be forwarded by the
Ryu-controlled Mininet switches. The physical experiment validates device
connectivity, local training, update transmission, and aggregation. The
software Mininet experiment separately validates the SDN forwarding behavior.

The first physical prototype aggregates the two physical client updates at the
laptop. It does not, by itself, prove a complete six-client physical hierarchical
SDN network or an accuracy improvement from S3TO.
