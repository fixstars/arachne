#! /bin/bash
set -euo pipefail
script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
common_dir=${script_dir}/../common/
sudo apt install -y gfortran libopenblas-dev liblapack-dev

# clone tvm
arachne_home=${script_dir}/../../
git clone --recursive https://github.com/fixstars/tvm ${arachne_home}/3rdparty/tvm

# install poetry
source ${common_dir}/install_poetry.sh
sudo apt-get install -y libhdf5-dev
# create virtual env
RUNTIME_ENV_DIR=${script_dir}
## download onnxruntime-gpu wheel
mkdir -p ${RUNTIME_ENV_DIR}/wheel
wget https://nvidia.box.com/shared/static/49fzcqa1g4oblwxr3ikmuvhuaprqyxb7.whl -O ${RUNTIME_ENV_DIR}/wheel/onnxruntime_gpu-1.6.0-cp36-cp36m-linux_aarch64.whl
## avoid numpy include failure
sudo ln -sf /usr/include/locale.h /usr/include/xlocale.h
cd ${RUNTIME_ENV_DIR}
poetry config virtualenvs.in-project true
poetry install
source ${RUNTIME_ENV_DIR}/.venv/bin/activate
## install tensorflow (Avoid the problem that h5py fails to install when using poetry)
pip install https://developer.download.nvidia.com/compute/redist/jp/v45/tensorflow/tensorflow-2.5.0+nv21.6-cp36-cp36m-linux_aarch64.whl

# build tvm
source ${common_dir}/install_tvm.sh ${script_dir}/../tvm_config/jetson_config.cmake
