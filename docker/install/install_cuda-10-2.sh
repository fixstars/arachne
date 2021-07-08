#!/bin/bash

set -euo pipefail

apt-get install -y --no-install-recommends cuda-runtime-10-2
apt-get install -y cuda-10.2
