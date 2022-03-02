#! /bin/bash
set -euo pipefail
script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
common_dir=${script_dir}/../common/
sudo apt install -y python3-pip gfortran libopenblas-dev liblapack-dev

# install poetry
source ${common_dir}/install_poetry.sh
sudo apt-get install -y libhdf5-dev

# create virtual env
RUNTIME_ENV_DIR=${script_dir}
## download onnxruntime-gpu wheel
mkdir -p ${RUNTIME_ENV_DIR}/wheel
wget https://nvidia.box.com/shared/static/jy7nqva7l88mq9i8bw3g3sklzf4kccn2.whl -O ${RUNTIME_ENV_DIR}/wheel/onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
## avoid numpy include failure
sudo ln -sf /usr/include/locale.h /usr/include/xlocale.h
cd ${RUNTIME_ENV_DIR}
poetry config virtualenvs.in-project true
poetry install
source ${RUNTIME_ENV_DIR}/.venv/bin/activate

# build tvm
source ${common_dir}/install_tvm.sh ${script_dir}/../tvm_config/jetson_config.cmake

