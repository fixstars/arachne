#!/bin/bash
set -euo pipefail

sudo apt-get install -y python3-pip python3-venv
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 - --version 1.2.0a2
export PATH="~/.local/bin:$PATH"
