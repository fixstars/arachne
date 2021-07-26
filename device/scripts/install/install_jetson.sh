#!/bin/bash

set -euo pipefail

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)

sudo apt install -y python3-pip gfortran libopenblas-dev liblapack-dev

python3 -m pip install --upgrade cython
python3 -m pip install scipy

source ${script_dir}/install_common.sh ${script_dir}/tvm_config/jetson_config.cmake
