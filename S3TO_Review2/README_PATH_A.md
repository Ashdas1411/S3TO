# S3TO Path A: Direct FedAvg vs. Hierarchical S3TO

## Purpose

This additive patch provides a controlled Mininet comparison between:

- **Hierarchical S3TO:** six clients send model updates to two assigned edge aggregators; each edge sends one sample-weighted aggregate to the global aggregator.
- **Direct FedAvg:** the same six clients send updates directly to the global aggregator; edge aggregation is bypassed.

The original default remains `--aggregation-mode hierarchical`. Existing benchmark JSON/CSV files are not overwritten by the patch. Run both new experiments with distinct output filenames.

## Experimental controls

The runner records and the comparison script validates:

- identical model name and FL configuration (seed, partition mode, sample count, local epochs, batch size, learning rate, rounds, and device);
- exact per-client training-example fingerprints, including the transformed image tensor values, labels, and selected indices;
- an SHA-256 fingerprint of the initial model weights;
- final and per-round accuracy, round wall-clock time, global-host-link counters, and an IPv4 forwarding byte-hop load proxy;
- that the global aggregator received the expected number of samples in every round.

The runner re-seeds immediately before initial model creation to avoid setup code consuming random numbers and silently changing the starting weights. It hashes a per-client fingerprint of the exact selected inputs/labels and the initial model. `scripts/compare_pathA_results.py` refuses to compare runs if the data partitions, initial weights, or core configuration differ.

## Network metrics and limitations

`network_metrics.global_host_link_at_s2_port5` reports rule counters for the specific switch path adjacent to the global host. These counters include TCP/IP headers, acknowledgments, and other network overhead; they are not application-payload-only bytes. The runner currently measures client/edge model-update transport and its responses; it does **not** transmit global model weights back to the client namespaces over Mininet. Therefore, these are not a complete count of every possible federated-learning uplink and downlink byte, and `responses_from_global` must not be described as a full global-model download.

`ipv4_forwarding_egress_byte_hops` sums active IPv4 forwarding-rule counters across physical switch egress ports. S3TO and direct mode both install explicit forward and reverse routes for their model-update/response paths. A packet traversing multiple switch hops is counted once per traversed hop. It is a **network-load proxy**, not unique end-to-end bytes. The raw sum across all OpenFlow entries is also retained but explicitly labelled as non-unique and should not be reported as total unique traffic. Before each mode is configured, the controller removes IPv4 rules matching all client, edge, and global endpoint addresses to prevent stale rules from contaminating later runs.

No performance benefit is assumed. S3TO could reduce bytes on the global-host link yet increase total byte-hops or wall-clock time. The experimental result—not the architecture's intended behavior—must determine the conclusion.

## Security changes

- Replaces pickle-based socket message framing with bounded JSON/tensor framing; the wire format uses a 4-byte length prefix and caps each decoded message at 64 MiB.
- Restricts the OS-Ken UDP control API to `127.0.0.1:9999`.
- Binds Mininet aggregator services to their specific emulated-host IP addresses and validates client/edge peer addresses and client assignments.
- Adds strict FedAvg structure, tensor-shape/dtype, and sample-count validation.
- Limits active TCP handlers and connection timeouts; the physical receiver defaults to loopback and requires an IP allowlist when binding to a non-loopback address. Temporary model checkpoint files are held in a run-specific private temporary directory and removed at exit.

**Security boundary:** these changes remove unsafe pickle deserialization from the active TCP message path and reduce accidental network exposure, but they do **not** implement TLS or cryptographic client authentication. Run this experiment only in a trusted, isolated lab network. IP allowlists are not a substitute for cryptographic authentication on an untrusted network; do not expose the services to the public internet.

**Existing Raspberry Pi bundles:** previously generated `results/hardware/pi1_bundle`, `pi2_bundle`, and tarballs contain the old client protocol. They must be regenerated to include `scripts/physical_fl_client.py` and `scripts/secure_wire.py` before future hardware tests. Do not use the older tarballs with the patched receiver.

## Apply safely

1. Copy `s3to_pathA_patch.zip` to `C:\Users\arkad\s3to_pathA_patch.zip` in Windows.
2. In Ubuntu/WSL, run:

```bash
set -euo pipefail
PY=/home/arkad/s3to/.venv/bin/python
PATCH_ZIP=/mnt/c/Users/arkad/s3to_pathA_patch.zip
[ -f "$PATCH_ZIP" ] || { echo "Patch ZIP not found: $PATCH_ZIP"; exit 1; }
UNPACK="$(mktemp -d /tmp/s3to_pathA_patch.XXXXXX)"
"$PY" - "$PATCH_ZIP" "$UNPACK" <<'PY'
import pathlib, sys, zipfile
archive = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2]).resolve()
with zipfile.ZipFile(archive) as bundle:
    for member in bundle.infolist():
        name = pathlib.PurePosixPath(member.filename)
        if name.is_absolute() or ".." in name.parts:
            raise SystemExit(f"Unsafe archive path: {member.filename!r}")
        target = (destination / pathlib.Path(*name.parts)).resolve()
        if target != destination and destination not in target.parents:
            raise SystemExit(f"Archive path escapes destination: {member.filename!r}")
    bundle.extractall(destination)
print("Patch archive extracted safely.")
PY
bash "$UNPACK/apply_pathA_patch.sh" "$HOME/s3to-github-copy/S3TO_Review2"
```

The installer checks SHA-256 hashes for every pre-existing source file it will replace. If any differs from the uploaded diagnostic snapshot, or if any new target already exists, it aborts before changing files. Existing targets are copied to a timestamped `/tmp/s3to_pathA_backup.*` directory. The installer does not run Git commands and does not reset/clean the repository.

## Validate after install

Run from the target repository in WSL:

```bash
cd ~/s3to-github-copy/S3TO_Review2 || exit 1
PY=/home/arkad/s3to/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1

"$PY" - <<'PY'
from pathlib import Path
bad = []
for root in (Path("common"), Path("fl"), Path("network"), Path("s3to"),
             Path("controller"), Path("scripts"), Path("tests")):
    if root.exists():
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                compile(path.read_text(encoding="utf-8"), str(path), "exec")
            except Exception as exc:
                bad.append((str(path), repr(exc)))
for path in (Path("run_s3to_mininet_fl.py"),):
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    except Exception as exc:
        bad.append((str(path), repr(exc)))
if bad:
    for item in bad:
        print(*item)
    raise SystemExit(1)
print("Syntax check passed (no bytecode files written).")
PY

"$PY" -m pytest -q tests
```

Do not start a new controller if port 6653 or UDP 9999 is already occupied. Inspect the process first. The code bundle was compile-checked and its tests passed in the patch-build environment, but that environment did not have Mininet or OS-Ken installed, so this is **not** a claim that end-to-end Mininet integration was run here.

## Run the controlled 10-round comparison

Only after validation passes, and only if no controller is already listening on these ports, start the patched OS-Ken controller from the repo root in WSL:

```bash
cd ~/s3to-github-copy/S3TO_Review2 || exit 1
set -euo pipefail
if ss -ltnp 2>/dev/null | grep -qE ':(6653)\b'; then
  echo "TCP port 6653 is already in use; do not start a duplicate controller. Inspect it first."
  exit 2
fi
if ss -lunp 2>/dev/null | grep -qE ':(9999)\b'; then
  echo "UDP port 9999 is already in use; do not start a duplicate controller. Inspect it first."
  exit 2
fi
mkdir -p results
nohup env PYTHONPATH="$PWD" "$HOME/s3to/.osken-venv/bin/osken-manager" \
  controller.s3to_osken --ofp-tcp-listen-port 6653 \
  > results/osken_controller_pathA.log 2>&1 < /dev/null &
echo $! > results/osken_controller_pathA.pid
sleep 3
ss -ltnp | grep -E ':(6653)\b'
ss -lunp | grep -E ':(9999)\b'
```

Run the same-seed smoke and 10-round experiments sequentially. The first pair is one round and acts as a go/no-go check; if either fails, the script stops before running the longer pair. If the smoke pair succeeds, the script runs one 10-round S3TO trial and one 10-round direct baseline. Outputs are timestamped and then bundled under `C:\Users\arkad\` for upload.

```bash
cd ~/s3to-github-copy/S3TO_Review2 || exit 1
set -euo pipefail
PY=/home/arkad/s3to/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
STAMP="$(date +%Y%m%d_%H%M%S)"
DATA=/home/arkad/s3to/data
CONFIG=config_fixed_local5.yaml
BASE_ARGS=(--config "$CONFIG" --no-download --require-real-mnist --model mlp \
  --data-dir "$DATA" --device cpu --controller-ip 127.0.0.1 \
  --controller-port 6653 --telemetry-port 9999)

# One-round go/no-go test. Both must complete before the 10-round runs begin.
sudo -E env PYTHONPATH="$PWD" "$PY" run_s3to_mininet_fl.py \
  "${BASE_ARGS[@]}" --rounds 1 --aggregation-mode hierarchical \
  --output "results/pathA_hierarchical_smoke_${STAMP}.json"
sudo -E env PYTHONPATH="$PWD" "$PY" run_s3to_mininet_fl.py \
  "${BASE_ARGS[@]}" --rounds 1 --aggregation-mode direct \
  --output "results/pathA_direct_smoke_${STAMP}.json"

# Main comparable 10-round runs, same config, seed, model, data, and device.
HIER="results/pathA_hierarchical_seed42_10rounds_${STAMP}.json"
DIRECT="results/pathA_direct_seed42_10rounds_${STAMP}.json"
COMPARE="results/pathA_comparison_seed42_10rounds_${STAMP}.json"
sudo -E env PYTHONPATH="$PWD" "$PY" run_s3to_mininet_fl.py \
  "${BASE_ARGS[@]}" --rounds 10 --aggregation-mode hierarchical --output "$HIER"
sudo -E env PYTHONPATH="$PWD" "$PY" run_s3to_mininet_fl.py \
  "${BASE_ARGS[@]}" --rounds 10 --aggregation-mode direct --output "$DIRECT"
"$PY" scripts/compare_pathA_results.py \
  --hierarchical "$HIER" --direct "$DIRECT" --output "$COMPARE"

# Keep only the new experiment outputs in the handoff ZIP.
"$PY" - "$STAMP" "$HIER" "$DIRECT" "$COMPARE" <<'PY'
import pathlib, sys, zipfile
stamp, *files = sys.argv[1:]
target = pathlib.Path(f"/mnt/c/Users/arkad/pathA_results_{stamp}.zip")
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
    for name in files:
        path = pathlib.Path(name)
        archive.write(path, path.name)
print("Comparison archive:", target)
PY
```

Upload `C:\Users\arkad\pathA_results_<timestamp>.zip` here. A one-seed run is an initial controlled result, not a multi-seed performance claim; if the pipeline succeeds, repeat with other seeds before drawing broad conclusions.
