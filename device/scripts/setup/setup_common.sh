#!/bin/bash

set -eo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: . setup_common.sh <rpc_key>"
    exit 1
fi

rpc_tracker=${ARACHNE_DEPLOY_RPC_TRACKER:-dgx-s.fixstars.com:8889}
rpc_key=$1

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
tvm_dir=${script_dir}/../../../3rdparty/tvm
build_dir=${script_dir}/build

export LD_LIBRARY_PATH=${build_dir}:$LD_LIBRARY_PATH
export PYTHONPATH=${tvm_dir}/python:$PYTHONPATH
python3 -m tvm.exec.rpc_server --tracker=${rpc_tracker} --key=${rpc_key} &
