#!/usr/bin/env bash

#
# Start a bash, mount /workspace to be current directory.
#
# Usage: docker/run.sh <CONTAINER_NAME>
#     Starts an interactive session
#
# Usage2: docker/run.sh <CONTAINER_NAME> [COMMAND]
#     Execute command in the docker image, non-interactive
#
if [ "$#" -lt 1 ]; then
    echo "Usage: docker/run.sh <IMAGE_NAME> <CONTAINER_NAME> [COMMAND]"
    exit -1
fi

DOCKER_IMAGE_NAME=("$1")
CONTAINER_NAME=("$2")

if [ "$#" -eq 2 ]; then
    COMMAND="bash"
    if [[ $(uname) == "Darwin" ]]; then
        # Docker's host networking driver isn't supported on macOS.
        # Use default bridge network and expose port for jupyter notebook.
        CI_DOCKER_EXTRA_PARAMS=("-it -p 8888:8888")
    else
        CI_DOCKER_EXTRA_PARAMS=("-it --net=host")
    fi
else
    shift 1
    COMMAND=("$@")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(pwd)"
DATASET_DIR=/raid/datasets


docker run --rm --pid=host \
    -v ${WORKSPACE}:/workspace \
    -v ${DATASET_DIR}:/datasets \
    -v ${SCRIPT_DIR}:/docker \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /dev/shm:/dev/shm -v /opt/xilinx/dsa:/opt/xilinx/dsa -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
    -w /workspace \
    -e "HOST_USER=$(id -u -n)" \
    -e "HOST_UID=$(id -u)" \
    -e "HOST_GROUP=$(id -g -n)" \
    -e "HOST_GID=$(id -g)" \
    -e "PYTHONPATH=/workspace/python" \
    --privileged \
    --gpus all \
    --name ${CONTAINER_NAME} \
    ${CI_DOCKER_EXTRA_PARAMS} \
    ${DOCKER_IMAGE_NAME} \
    bash --login /docker/with_the_same_user.sh \
    ${COMMAND[@]}
