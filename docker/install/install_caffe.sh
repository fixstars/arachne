#!/bin/bash

set -euo pipefail

apt-get install -y tzdata
apt-get install -y caffe-cpu

python3 -m pip install --upgrade scikit-image
