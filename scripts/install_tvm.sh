#! /bin/bash

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)

TVM_SOURCE_DIR=${script_dir}/../3rdparty/tvm
BUILD_DIR=${script_dir}/../build/tvm

mkdir -p ${BUILD_DIR}
cp ${TVM_SOURCE_DIR}/cmake/config.cmake ${BUILD_DIR}/config.cmake
echo "set(USE_LLVM llvm-config-11)" >> ${BUILD_DIR}/config.cmake
echo "set(USE_BLAS openblas)" >> ${BUILD_DIR}/config.cmake
echo "set(USE_CUDA /usr/local/cuda)" >> ${BUILD_DIR}/config.cmake
echo "set(USE_CUDNN ON)" >> ${BUILD_DIR}/config.cmake
echo "set(USE_CUBLAS ON)" >> ${BUILD_DIR}/config.cmake
echo "set(USE_TENSORRT_CODEGEN ON)" >> ${BUILD_DIR}/config.cmake
echo "set(USE_TENSORRT_RUNTIME ON)" >> ${BUILD_DIR}/config.cmake

cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -S ${TVM_SOURCE_DIR} -B ${BUILD_DIR}
cmake --build ${BUILD_DIR}

cd ${TVM_SOURCE_DIR}/python
python -m pip install -e .