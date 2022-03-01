#!/bin/bash

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: . setup.sh <env_dirname> <runtime_name> <port>"
    exit 1
fi
env_dirname = $1
runtime_name=$2
rpc_port=$3
script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
tvm_dir=${script_dir}/../3rdparty/tvm
build_dir=${script_dir}/../build/tvm
arachne_root=${script_dir}/..

source ${script_dir}/${env_dirname}/.venv/bin/activate
export LD_LIBRARY_PATH=${build_dir}:$LD_LIBRARY_PATH
export PYTHONPATH=${tvm_dir}/python:${arachne_root}/python:$PYTHONPATH
export PATH=/usr/local/cuda/bin:${PATH}
python -m arachne.runtime.rpc.server --port ${rpc_port} --runtime ${runtime_name}
