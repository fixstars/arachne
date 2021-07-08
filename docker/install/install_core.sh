#!/bin/bash

set -euo pipefail

apt-get install -y git 

# install libraries for building c/c++ core on ubuntu
apt-get update --fix-missing

apt-get install -y --no-install-recommends \
    git-lfs build-essential cmake ninja-build libgtest-dev wget unzip curl libtinfo-dev libz-dev \
    libgnome-keyring-dev libopenblas-dev sudo nodejs npm \
    apt-transport-https graphviz doxygen gpg-agent libprotobuf-dev protobuf-compiler

# install liuraries for cross-compiling
apt-get install -y gcc-multilib g++-multilib
apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu

cd /usr/src/gtest && cmake CMakeLists.txt && make && cp *.a /usr/lib
