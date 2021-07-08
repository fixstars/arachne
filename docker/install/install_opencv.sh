#!/bin/bash

set -euo pipefail

git clone --branch=4.5.2 --depth 1 --recursive https://github.com/opencv/opencv.git
git clone --branch=4.5.2 --depth 1 --recursive https://github.com/opencv/opencv_contrib.git

cd opencv
mkdir -p build && cd build
# TODO(Maruoka): Enable CUDA
cmake -GNinja \
    -D CMAKE_BUILD_TYPE=Release \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_opencv_python3=ON \
    ../
    #-D CMAKE_EXTRA_MODULES_PATH=../../opencv_contrib/modules
    # -D WITH_OPENMP=ON \
    #-D WITH_CUDA=ON \
    #-D WITH_CUBLAS=ON \
    #-D HAVE_opencv_cudev=ON \
    #-D OPENCV_DNN_CUDA=ON 

# install
cmake --build . --target install
