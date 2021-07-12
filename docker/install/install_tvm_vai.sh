#!/bin/bash

set -euo pipefail

cd /tvm

mkdir -p build && cd build

# CMake configrations
cp /tvm/cmake/config.cmake .
echo set\(USE_LLVM ON\) >> config.cmake
echo set\(USE_VITIS_AI ON\) >> config.cmake

# Build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..
cmake --build .

# Install pip package
cd /tvm/python
python3 -m pip install -e .
