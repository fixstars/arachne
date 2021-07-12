#!/bin/bash

set -euo pipefail

apt-get install -y software-properties-common
apt-get install -y python3-dev python3-cairocffi python3-pip

python3 -m pip install -U pip
