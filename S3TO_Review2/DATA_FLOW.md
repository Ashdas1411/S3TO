# S3TO Data Flow for Review 2

1. Each FL client trains from a local Non-IID partition.
2. The client computes a label-distribution histogram.
3. The histogram is sent as lightweight UDP telemetry to the S3TO controller.
4. The controller computes pairwise Cosine Similarity and forms two complementary client groups.
5. Each client is assigned to edge1 or edge2.
6. Model updates are sent using TCP to the selected edge aggregator.
7. Each edge aggregator performs FedAvg over its assigned clients.
8. Edge models are sent to the global aggregator using TCP.
9. The global aggregator performs another weighted FedAvg and produces the global model for the next round.
10. In the Mininet/Ryu layer, the same assignment is translated into OpenFlow 1.3 forwarding rules.

## Hardware-ready extension

Physical Raspberry Pi/ESP32 clients can emit the same telemetry structure later. Optional link-quality values can be added without changing the S3TO core message schema.
