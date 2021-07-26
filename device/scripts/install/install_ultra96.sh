#!/bin/bash
set -euo pipefail

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)

git clone --branch v1.2.0 --recursive --shallow-submodules https://github.com/Xilinx/DPU-PYNQ.git
cd DPU-PYNQ/upgrade
make
sudo pip3 install pynq-dpu==1.2.0
sudo python3 -c 'from pynq_dpu import DpuOverlay ; overlay = DpuOverlay("dpu.bit")'

sudo apt-get install libhdf5-dev
sudo pip3 install pydot==1.4.1 h5py==2.8.0

cd ../../
git clone --recursive https://github.com/Xilinx/pyxir.git
cd pyxir
sudo python3 setup.py install --use_vai_rt_dpuczdx8g

source ${script_dir}/install_common.sh ${script_dir}/tvm_config/ultra96_config.cmake
