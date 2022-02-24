#! /bin/bash
set -euo pipefail
script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)

sudo apt install -y python3-pip gfortran libopenblas-dev liblapack-dev

# install poetry
sudo apt-get install python3-venv
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python3 - --version 1.2.0a2
export PATH="~/.local/bin:$PATH"

# create virtual env
RUNTIME_ENV_DIR=${script_dir}/env/
## download onnxruntime-gpu wheel
mkdir -p ${RUNTIME_ENV_DIR}/wheel
wget https://nvidia.box.com/shared/static/jy7nqva7l88mq9i8bw3g3sklzf4kccn2.whl -O ${RUNTIME_ENV_DIR}/wheel/onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
## avoid numpy include failure
sudo ln -s /usr/include/locale.h /usr/include/xlocale.h
cd ${RUNTIME_ENV_DIR}
poetry config virtualenvs.in-project true
poetry install
source ${RUNTIME_ENV_DIR}/.venv/bin/activate

# build tvm
TVM_SOURCE_DIR=${script_dir}/../3rdparty/tvm
BUILD_DIR=${script_dir}/../build/tvm

mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
cp ${script_dir}/tvm_config/jetson_config.cmake ${BUILD_DIR}/config.cmake

cmake -DCMAKE_BUILD_TYPE=Release ${TVM_SOURCE_DIR}
cmake --build . --target runtime
