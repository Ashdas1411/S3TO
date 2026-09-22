#!/usr/bin/env bash
set -e
sudo apt update
sudo apt install -y python3-venv python3-pip mininet openvswitch-switch openvswitch-common
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-ryu-mininet.txt
