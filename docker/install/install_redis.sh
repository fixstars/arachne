#!/bin/bash

set -euo pipefail

apt-get install -y redis-server
python3 -m pip install xgboost>=1.1.0 psutil
