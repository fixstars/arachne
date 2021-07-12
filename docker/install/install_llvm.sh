#!/bin/bash

set -euo pipefail

echo deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main\
     >> /etc/apt/sources.list.d/llvm.list
echo deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main\
     >> /etc/apt/sources.list.d/llvm.list

wget -q -O - http://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
apt-get update && apt-get install -y llvm-11 clang-11
