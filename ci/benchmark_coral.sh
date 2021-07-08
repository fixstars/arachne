#!/bin/bash

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: /ci/benchmark.sh <device_name>"
    exit 1
fi

device=$1

rpc_tracker=$ARACHNE_CI_RPC_TRACKER
rpc_key=$ARACHNE_CI_RPC_KEY

experiments=(
    "ssd_mobilenet_v1_coco"
    "ssd_mobilenet_v2_coco"
    "ssdlite_mobilenet_v2_coco"
    "ssdlite_mobiledet_edgetpu_coco"
    "ssdlite_mobiledet_dsp_coco"
    "ssdlite_mobiledet_gpu_coco" 
)

for experiment in ${experiments[@]} ; do
    python3 ci/benchmark_pipeline.py \
      --experiment=${experiment} \
      --device=${device} \
      --rpc-tracker=${rpc_tracker} \
      --rpc-key=${rpc_key}
done
