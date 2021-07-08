#!/bin/bash

set -euo pipefail

VERSION=2.4.1

# Install keras_preprocessing
python3 -m pip install keras_preprocessing --no-deps

# Clone tensorflow
git clone https://github.com/tensorflow/tensorflow -b v${VERSION} --depth 1
cd tensorflow

# Build tensorflow
TF_NEED_CUDA=1 TF_NEED_TENSORRT=1 ./configure
bazelisk build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
python3 -m pip install /tmp/tensorflow_pkg/tensorflow-${VERSION}-cp36-cp36m-linux_x86_64.whl

cd ..

# Install tfds
python3 -m pip install tensorflow-datasets
