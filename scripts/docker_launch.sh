#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run \
    --rm \
    -it \
    -u root \
    --gpus all \
    -v $(pwd):/workspaces/arachne \
    -w /workspaces/arachne \
    -e "HOST_UID=$(id -u)" \
    -e "HOST_GID=$(id -g)" \
    -e "PYTHONPATH=/workspaces/arachne/python" \
    -e "TVM_LIBRARY_PATH=/workspaces/arachne/build/tvm" \
    arachnednn/arachne:devel-gpu \
    bash /workspaces/arachne/scripts/_docker_init.sh