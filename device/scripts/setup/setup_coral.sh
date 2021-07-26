#!/bin/bash

set -eo pipefail

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)

export LD_LIBRARY_PATH=$HOME/.local/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

source ${script_dir}/setup_common.sh coral

