#!/bin/bash

set -euo pipefail

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)

curl -O https://ion-archives.s3-us-west-2.amazonaws.com/genesis-runtime/libedgetpu-release-frogfish-tf2.4.1/libedgetpu1-max_14.0_arm64.deb
curl -O https://ion-archives.s3-us-west-2.amazonaws.com/genesis-runtime/libedgetpu-release-frogfish-tf2.4.1/libedgetpu-dev_14.0_arm64.deb

mkdir -p ~/.local

dpkg -x libedgetpu1-max_14.0_arm64.deb ~/.local
dpkg -x libedgetpu-dev_14.0_arm64.deb ~/.local

export CMAKE_PREFIX_PATH=$HOME/.local/usr

source ${script_dir}/install_common.sh ${script_dir}/tvm_config/coral_config.cmake
