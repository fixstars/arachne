#!/bin/bash

set -euo pipefail

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)

sudo apt install -y build-essential cmake python3-pip

source ${script_dir}/install_common.sh ${script_dir}/tvm_config/raspi4_config.cmake
