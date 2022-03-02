#! /bin/bash
set -euo pipefail
script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
common_dir=${script_dir}/../common/
sudo apt install -y python3-pip gfortran libopenblas-dev liblapack-dev

# install poetry
source ${common_dir}/install_poetry.sh
sudo apt-get -y install libhdf5-dev

# create virtual env
RUNTIME_ENV_DIR=${script_dir}
## avoid numpy include failure
sudo ln -sf /usr/include/locale.h /usr/include/xlocale.h
cd ${RUNTIME_ENV_DIR}
poetry config virtualenvs.in-project true
poetry install
source ${RUNTIME_ENV_DIR}/.venv/bin/activate

# build tvm
source ${common_dir}/install_tvm.sh ${script_dir}/../tvm_config/raspi4_config.cmake

