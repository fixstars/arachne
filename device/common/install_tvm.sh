#!/bin/bash

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: . install_common.sh <config.cmake_file_name>"
    exit 1
fi

config_cmake_file=$1

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
tvm_dir=${script_dir}/../../3rdparty/tvm
build_dir=${tvm_dir}/build

mkdir -p ${build_dir} && cd ${build_dir}

# CMake configrations
cp ${config_cmake_file} config.cmake

# Build
cmake -DCMAKE_BUILD_TYPE=Release ${tvm_dir}
cmake --build . --target runtime
