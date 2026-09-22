#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
ryu-manager --observe-links --ofp-tcp-listen-port 6653 controller/s3to_ryu.py
