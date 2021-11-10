FROM gitlab.fixstars.com:5005/arachne/containers:v0.1.3

# Setting working directory
ARG workspace
ENV workspace /workspace
WORKDIR $workspace

# Environment variables
ENV TVM_LIBRARY_PATH=${workspace}/build_tvm
ENV TVM_DARKNET_LIB_DIR=/usr/lib

## for black
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Disable cache (#81)
ARG CACHEBUST=0
RUN :
