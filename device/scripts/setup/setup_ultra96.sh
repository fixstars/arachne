#!/bin/bash

set -eo pipefail

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)

source ${script_dir}/setup_common.sh ultra96
