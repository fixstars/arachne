FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update --fix-missing

COPY install/install_core.sh /install/install_core.sh
RUN bash /install/install_core.sh

COPY install/install_core_gpu.sh /install/install_core_gpu.sh
RUN bash /install/install_core_gpu.sh

COPY install/install_devtools.sh /install/install_devtools.sh
RUN bash /install/install_devtools.sh

COPY install/install_python.sh /install/install_python.sh
RUN bash /install/install_python.sh

COPY install/install_python_package.sh /install/install_python_package.sh
RUN bash /install/install_python_package.sh

COPY install/install_llvm.sh /install/install_llvm.sh
RUN bash /install/install_llvm.sh

COPY install/install_bazel.sh /install/install_bazel.sh
RUN bash /install/install_bazel.sh

COPY install/install_tensorrt.sh /install/install_tensorrt.sh
RUN bash /install/install_tensorrt.sh

# Install DNN frameworks
COPY install/install_onnx.sh /install/install_onnx.sh
RUN bash /install/install_onnx.sh

COPY install/install_torch.sh /install/install_torch.sh
RUN bash /install/install_torch.sh

COPY install/install_tensorflow.sh /install/install_tensorflow.sh
RUN bash /install/install_tensorflow.sh

COPY install/install_tflite.sh /install/install_tflite.sh
RUN bash /install/install_tflite.sh

COPY install/install_edgetpu.sh /install/install_edgetpu.sh
RUN bash /install/install_edgetpu.sh

COPY install/install_redis.sh /install/install_redis.sh
RUN bash /install/install_redis.sh

COPY install/install_universal.sh /install/install_universal.sh
RUN bash /install/install_universal.sh

COPY install/install_opencv.sh /install/install_opencv.sh
RUN bash /install/install_opencv.sh

COPY install/install_darknet.sh /install/install_darknet.sh
RUN bash /install/install_darknet.sh

COPY install/install_caffe.sh /install/install_caffe.sh
RUN bash /install/install_caffe.sh

COPY install/install_docker.sh /install/install_docker.sh
RUN bash /install/install_docker.sh

# Setting working directory
ARG workspace
ENV workspace /workspace
WORKDIR $workspace

# Environment variables
ENV PATH=/usr/local/nvidia/bin:${PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV C_INCLUDE_PATH=/usr/local/cuda/include:${C_INCLUDE_PATH}
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compact:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compact:${LD_LIBRARY_PATH}
ENV TVM_LIBRARY_PATH=${workspace}/build_tvm
ENV TF_CPP_MIN_LOG_LEVEL 3
ENV TVM_DARKNET_LIB_DIR=/usr/lib

# Disable cache (#81)
ARG CACHEBUST=0
RUN :