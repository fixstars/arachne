#!/bin/bash

set -euo pipefail

# NOTE: make sure that cmake >= 3.13, nvidia, and cuda are installed.

VERSION=1.8.1

python3 -m pip install "onnx==${VERSION}"
python3 -m pip install future

git clone --recursive https://github.com/Microsoft/onnxruntime -b v${VERSION} --depth 1
cd onnxruntime

# Start the basic build
./build.sh --config Release --update --build --build_shared_lib --enable_pybind --build_wheel --parallel --skip_tests --use_cuda --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu --tensorrt_home /usr/lib/x86_64-linux-gnu --cmake_extra_defines CMAKE_INSTALL_PREFIX=/usr/local/onnxruntime CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# install header files and shared libraries
cd ./build/Linux/Release && make install

# install python package
python3 -m pip install dist/*.whl
