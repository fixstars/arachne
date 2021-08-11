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
cp ${tvm_dir}/cmake/config.cmake .
echo set\(USE_CUDA /usr/local/cuda-10.2\) >> config.cmake
echo set\(USE_CPP_RPC ON\) >> config.cmake
echo set\(USE_GRAPH_EXECUTOR_CUDA_GRAPH ON\) >> config.cmake
echo set\(USE_LLVM llvm-config-11\) >> config.cmake
echo set\(USE_BLAS openblas\) >> config.cmake
echo set\(USE_TFLITE ON\) >> config.cmake
echo set\(USE_TENSORFLOW_PATH /tensorflow\) >> config.cmake
echo set\(USE_FLATBUFFERS_PATH /usr/local\) >> config.cmake
echo set\(USE_EDGETPU ON\) >> config.cmake
echo set\(USE_CUDNN ON\) >> config.cmake
echo set\(USE_CUBLAS ON\) >> config.cmake
echo set\(USE_TENSORRT_CODEGEN ON\) >> config.cmake
echo set\(USE_TENSORRT_RUNTIME ON\) >> config.cmake
echo set\(USE_ONNX_RUNTIME /usr/local/onnxruntime\) >> config.cmake

# Build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ${tvm_dir}
cmake --build .

# Install pip package
cd ${tvm_dir}/python
python3 -m pip install -e .
