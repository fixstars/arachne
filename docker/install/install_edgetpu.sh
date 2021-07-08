#!/bin/bash

set -euo pipefail

# edgetpu-compiler
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

apt-get update && apt-get install edgetpu-compiler libusb-1.0-0

# libedgetpu
curl -O https://ion-archives.s3-us-west-2.amazonaws.com/genesis-runtime/libedgetpu-release-frogfish-tf2.4.1/libedgetpu1-std_14.0_amd64.deb
curl -O https://ion-archives.s3-us-west-2.amazonaws.com/genesis-runtime/libedgetpu-release-frogfish-tf2.4.1/libedgetpu-dev_14.0_amd64.deb

dpkg -i libedgetpu1-std_14.0_amd64.deb
dpkg -i libedgetpu-dev_14.0_amd64.deb
