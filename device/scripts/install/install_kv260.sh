#!/bin/bash

set -euo pipefail

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)

# Install rpm packages
dnf repoquery
sudo dnf install -y \
    git \
    git-perltools \
    packagegroup-petalinux-self-hosted \
    cmake \
    dnndk \
    packagegroup-petalinux-vitisai-dev.noarch \
    python3-cached-property \
    python3-numpy \
    python3-h5py \
    python3-pydot \
    python3-pyparsing \

# Install pip packages
sudo python3 -m pip install cloudpickle

# Install PyXIR
git clone --recursive https://github.com/Xilinx/pyxir.git
cd pyxir
patch -p1 < ${script_dir}/pyxir.patch
sudo python3 setup.py install --use_vai_rt_dpuczdx8g

# Install TVM
source ${script_dir}/install_common.sh ${script_dir}/tvm_config/zynq_config.cmake
