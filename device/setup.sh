#!/bin/bash

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: . setup.sh <runtime_name> <port>"
    exit 1
fi
runtime_name=$1
rpc_port=$2
script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
tvm_dir=${script_dir}/../3rdparty/tvm
build_dir=${script_dir}/../build/tvm
arachne_root=${script_dir}/..

source ${script_dir}/env/.venv/bin/activate
pip list | grep tensorflow
export LD_LIBRARY_PATH=${build_dir}:$LD_LIBRARY_PATH
export PYTHONPATH=${tvm_dir}/python:${arachne_root}/python:$PYTHONPATH
echo $PYTHONPATH
python -m arachne.runtime.rpc.server --port ${rpc_port} --runtime ${runtime_name}
