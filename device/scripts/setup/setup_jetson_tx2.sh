#!/bin/bash

set -eo pipefail

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)

export TVM_TENSORRT_USE_FP16=1
# export TVM_TENSORRT_CACHE_DIR=/tmp

source ${script_dir}/setup_common.sh jetson-tx2
