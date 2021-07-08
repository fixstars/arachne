#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: /ci/evaluate.sh <device_name>"
    exit 1
fi

device=$1
sample_arg=""
if [ $# -ge 2 ]; then
    sample_arg="--evaluate-sample=$2"
fi

experiments=(
    "ssd_mobilenet_v1_coco"
    "ssd_mobilenet_v2_coco"
    "ssdlite_mobilenet_v2_coco"
    "ssd_mobilenet_v3_small_coco"
    "ssd_mobilenet_v3_large_coco"
    "ssdlite_mobiledet_cpu_coco"
    "ssdlite_mobiledet_edgetpu_coco"
    "ssdlite_mobiledet_gpu_coco" 
    "mobilenet_v2_imagenet"
    "mobilenet_v3_small_imagenet"
    "mobilenet_v3_large_imagenet"
    "yolov3_coco"
    "yolov3_tiny_coco"
)

for experiment in ${experiments[@]} ; do
    python3 ci/benchmark_pipeline.py \
      --experiment=${experiment} \
      --device=${device} \
      --disable-benchmark \
      --evaluate \
      ${sample_arg}
done
