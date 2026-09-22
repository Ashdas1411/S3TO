# S3TO Review-2 Prototype

**Statistical-Skew-Driven SDN Topology Orchestration for Hierarchical Federated Learning**

This repository is intentionally split into two layers:

1. **Fully integrated local prototype** — runs with ordinary Python sockets. It demonstrates Non-IID FL, UDP label-distribution telemetry, Cosine Similarity grouping, TCP model-update transport, two edge aggregators, and one global aggregator. This is the fastest way to get a working end-to-end demo.
2. **Mininet + Ryu SDN prototype** — runs in WSL/Ubuntu. It demonstrates the same S3TO decision being applied as OpenFlow forwarding rules on a 3-switch topology.

The hardware/RSSI part is represented by the telemetry schema's optional `link_quality` field so physical clients can be added later without changing the core message format.

## Folder placement

Put the whole `S3TO_Review2` folder somewhere convenient, for example:

```text
C:\Users\<your-name>\Desktop\S3TO_Review2
```

For Mininet/Ryu, copy or clone the same folder into WSL, for example:

```text
~/S3TO_Review2
```

## 1. Fastest working demo

Use Python 3.10/3.11 if possible.

```bash
cd S3TO_Review2
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# WSL/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

First test only the S3TO algorithm:

```bash
python run_s3to_algorithm.py
```

Then run the complete local hierarchical FL + UDP + TCP demo:

```bash
python run_s3to_local.py --rounds 2
```

For a no-download smoke test:

```bash
python run_s3to_local.py --rounds 1 --no-download
```

Files generated:

```text
results/s3to_algorithm_result.json
results/s3to_local_results.json
```

## 2. Baseline FedAvg

```bash
python run_fl_baseline.py --rounds 2
```

Output:

```text
results/baseline_results.json
```

After both runs:

```bash
python experiments/compare.py
```

This creates:

```text
results/baseline_vs_s3to_accuracy.png
```

Do not present the generated accuracy values as final research results until you have run repeated controlled experiments. They are development/demo results.

## 3. Mininet + Ryu demo

Use WSL/Ubuntu with root privileges.

A convenient setup is:

```bash
cd ~/S3TO_Review2
bash scripts/setup_ubuntu.sh
source .venv/bin/activate
```

Then terminal 1:

```bash
cd ~/S3TO_Review2
source .venv/bin/activate
bash scripts/start_ryu.sh
```

Terminal 2:

```bash
cd ~/S3TO_Review2
source .venv/bin/activate
sudo -E .venv/bin/python run_mininet_demo.py --cli
```

The topology is:

```text
c1 c2 ---- s1 ---- s2 ---- s3 ---- c5 c6
             |        |        |
           edge1     c3 c4   edge2
```

The Ryu application listens for UDP commands on port `9999` and installs deterministic OpenFlow rules for the fixed Mininet topology. Inspect them with:

```bash
sudo ovs-ofctl -O OpenFlow13 dump-flows s1
sudo ovs-ofctl -O OpenFlow13 dump-flows s2
sudo ovs-ofctl -O OpenFlow13 dump-flows s3
```

## 4. What the code demonstrates

### FL layer
- PyTorch MLP for MNIST
- deterministic label-skewed Non-IID partitions
- local client training
- FedAvg
- two-level hierarchical aggregation

### S3TO layer
- per-client label histogram
- Cosine Similarity
- enumeration of 3+3 client partitions
- selection of the partition with low within-group similarity
- mapping of groups to two edge aggregators

### Networking layer
- UDP for small telemetry/control messages
- TCP for model-update transport
- Mininet topology
- Ryu SDN controller
- OpenFlow 1.3 flow rules

### Hardware-ready interface
The telemetry structure already permits optional:

```json
"link_quality": {
  "rssi_dbm": -61,
  "snr_db": 22
}
```

No physical hardware is required to run the Review-2 software demonstration.

## 5. Important development notes

- The local integrated demo is deliberately the fastest end-to-end path.
- The Mininet/Ryu portion is a separate SDN emulation layer because Mininet requires Linux/root privileges.
- Keep the hardware/RSSI extension out of the core Review-2 algorithm unless your team deliberately expands scope later.
- Keep a Git commit after each working milestone so Review 2 has visible evidence of progress.

Suggested commits:

```text
feat: add non-iid mnist partitioning
feat: add fedavg baseline
feat: add s3to cosine similarity grouping
feat: add udp telemetry controller
feat: add tcp hierarchical aggregation
feat: add ryu openflow controller
feat: add mininet topology
feat: add baseline vs s3to experiment
```
