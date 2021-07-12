#!/bin/bash

set -euo pipefail

os="ubuntu1804"
cuda="cuda10.2"
trt="7.2.3"
tag="${cuda}-trt${trt}.4-ga-20210226"
deb="nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb"
baseurl="https://ion-archives.s3-us-west-2.amazonaws.com/genesis-runtime"
ver="${trt}-1+${cuda}"

apt-get install -y libnvinfer7=${ver} libnvonnxparsers7=${ver} libnvparsers7=${ver} libnvinfer-plugin7=${ver} libnvinfer-dev=${ver} libnvonnxparsers-dev=${ver} libnvparsers-dev=${ver} libnvinfer-plugin-dev=${ver} python-libnvinfer=${ver} python3-libnvinfer=${ver}

curl -O ${baseurl}/${deb}
dpkg -i ${deb}
apt-key add /var/nv-tensorrt-repo-${os}-${tag}/7fa2af80.pub
apt-get update && apt-get install -y tensorrt
