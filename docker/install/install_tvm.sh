#!/bin/bash

set -euo pipefail

current_dir=$(cd $(dirname $0); pwd)
script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)

if [ -z ${TVM_DIR+x} ];
then
    tvm_dir=${script_dir}/../../3rdparty/tvm
else
    tvm_dir=${TVM_DIR}
fi
mkdir -p build_tvm && cd build_tvm

# CMake configrations
${script_dir}/setup_tvm_config.sh

# Build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ${tvm_dir}
cmake --build .

# Install pip package
cd ${tvm_dir}/python
python3 -m pip install -e .
