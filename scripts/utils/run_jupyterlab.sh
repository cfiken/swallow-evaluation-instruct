#!/bin/bash

PROJECT_DIR=$(dirname $(realpath $0))
set -a; source "${PROJECT_DIR}/../../.env"; set +a;

port=$(python -c $'import socket\nwith socket.socket() as s:\n    s.bind(("", 0))\n    print(s.getsockname()[1])')
uv run --isolated --locked --project ${PROJECT_DIR}/pyproject.toml jupyter lab --port $port