#!/bin/bash

set -euo pipefail

# Download, build and install flatbuffers
git clone --branch=v1.12.0 --depth=1 --recursive https://github.com/google/flatbuffers.git
cd flatbuffers
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make install -j8
cd ..

# Install flatbuffers python packages.
python3 -m pip install flatbuffers

# Build tensorflow-lite
apt-get install -y swig libjpeg-dev zlib1g-dev python3-dev python3-numpy
python3 -m pip install numpy pybind11

/tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
/tensorflow/tensorflow/lite/tools/make/build_lib.sh

# Setup tflite from schema
mkdir -p tflite && cd tflite
cp /tensorflow/tensorflow/lite/schema/schema.fbs .
flatc --python schema.fbs

cat <<EOM >setup.py
import setuptools
setuptools.setup(
    name="tflite",
    version="2.3.2",
    author="google",
    author_email="google@google.com",
    description="TFLite",
    long_description="TFLite",
    long_description_content_type="text/markdown",
    url="https://www.tensorflow.org/lite",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
EOM

cat <<EOM >__init__.py
name = "tflite"
EOM

# Install tflite over python3
python3 setup.py install

cd ..
