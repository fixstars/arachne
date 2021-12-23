# Copyright 2021 Takafumi Kubota. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

ARG UBUNTU_VERSION=18.04

ARG ARCH=
ARG CUDA=10.2
ARG CUDNN_MAJOR_VERSION=8
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-cudnn${CUDNN_MAJOR_VERSION}-devel-ubuntu${UBUNTU_VERSION}
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG UBUNTU_VERSION
ARG ARCH
ARG CUDA
ARG CUDNN_MAJOR_VERSION

# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

# Install Cuda and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cuda-command-line-tools-${CUDA} \
    cuda-nvrtc-${CUDA/./-} \
    cuda-nvrtc-dev-${CUDA/./-} \
    curl \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libzmq3-dev \
    pkg-config \
    software-properties-common \
    unzip

RUN apt-get update && \
    if [[ ${CUDA} == *"10."* ]]; then\
        apt-get install -y --no-install-recommends \
        libcublas10 \
        libcublas-dev; \
    else \
        apt-get install -y --no-install-recommends \
        libcublas-${CUDA/./-} \
        libcufft-${CUDA/./-} \
        libcurand-${CUDA/./-} \
        libcusolver-${CUDA/./-} \
        libcusparse-${CUDA/./-}; \
    fi


# Install TensorRT
ARG LIBNVINFER=7.2.3-1
ARG LIBNVINFER_MAJOR_VERSION=7
ENV LIBNVINFER_APT_VER ${LIBNVINFER}+cuda${CUDA}
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
        apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER_APT_VER} \
        libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER_APT_VER} \
        libnvparsers${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER_APT_VER} \
        libnvonnxparsers${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER_APT_VER} \
        python3-libnvinfer=${LIBNVINFER_APT_VER} \
        python3-libnvinfer-dev=${LIBNVINFER_APT_VER} \
        libnvinfer-dev=${LIBNVINFER_APT_VER} \
        libnvinfer-plugin-dev=${LIBNVINFER_APT_VER} \
        libnvparsers-dev=${LIBNVINFER_APT_VER} \
        libnvonnxparsers-dev=${LIBNVINFER_APT_VER} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }

## TODO host other versions
# RUN curl -O https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/nv-tensorrt-repo-ubuntu${UBUNTU_VERSION/./}-cuda${CUDA}-trt${LIBNVINFER/-1/}.4-ga-20210226_1-1_amd64.deb

# RUN dpkg -i nv-tensorrt-repo-ubuntu${UBUNTU_VERSION/./}-cuda${CUDA}-trt${LIBNVINFER/-1/}.4-ga-20210226_1-1_amd64.deb \
#     && apt-key add /var/nv-tensorrt-repo-ubuntu${UBUNTU_VERSION/./}-cuda${CUDA}-trt${LIBNVINFER/-1/}.4-ga-20210226/7fa2af80.pub \
#     && apt-get update \
#     && apt-get install -y tensorrt \
#     && rm nv-tensorrt-repo-ubuntu${UBUNTU_VERSION/./}-cuda${CUDA}-trt${LIBNVINFER/-1/}.4-ga-20210226_1-1_amd64.deb


# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda-${CUDA}/targets/x86_64-linux/lib:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install llvm-11
RUN echo deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main >> /etc/apt/sources.list.d/llvm.list \
    && echo deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main >> /etc/apt/sources.list.d/llvm.list \
    && apt-key adv --fetch-keys http://apt.llvm.org/llvm-snapshot.gpg.key \
    && apt-get update && apt-get install -y llvm-11 clang-11

RUN apt-get update && apt-get install -y libopenblas-dev

# Install other packages for development
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    sudo \
    git

# python -> python3
RUN ln -s $(which python3) /usr/local/bin/python

# Add a user that UID:GID will be updated by vscode
ARG USERNAME=developer
ARG GROUPNAME=develoepr
ARG UID=1000
ARG GID=1000
ARG PASSWORD=developer
RUN groupadd -g $GID $GROUPNAME && \
    useradd -m -s /bin/bash -u $UID -g $GID -G sudo $USERNAME && \
    echo $USERNAME:$PASSWORD | chpasswd && \
    echo "$USERNAME   ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USERNAME

# install poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

