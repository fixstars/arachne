#!/bin/sh
REPO_URL="https://github.com/dmlc/web-data/blob/main/darknet/"
if [[ "$(uname)" == 'Darwin' ]]; then
    DARKNET_LIB="libdarknet_mac2.0.so"
    DARKNET_URL=$REPO_URL"lib_osx/"$DARKNET_LIB"?raw=true"

elif [[ "$(expr substr $(uname -s) 1 5)" == 'Linux' ]]; then
    DARKNET_LIB="libdarknet2.0.so"
    DARKNET_URL=$REPO_URL"lib/"$DARKNET_LIB"?raw=true"
fi
TVM_DARKNET_LIB_PATH="/usr/lib/"$DARKNET_LIB
wget $DARKNET_URL -O $TVM_DARKNET_LIB_PATH