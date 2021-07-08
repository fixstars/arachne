#!/bin/bash

set -euo pipefail

current_dir=$(cd $(dirname $0); pwd)
script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
git clone --recursive https://github.com/Xilinx/pyxir

#modify ultra96 hardware file for DPU-PYNQ v2.6
wget https://www.xilinx.com/bin/public/openDownload?filename=pynqdpu.dpu.ultra96.hwh -O ultra96.hwh
dlet -f ultra96.hwh
mv `ls -t *.dcf | head -n1` Ultra96.dcf
echo -e "{\n\t\"target\"   : \"dpuv2\",\n\t\"dcf\"      : \"Ultra96.dcf\",\n\t\"cpu_arch\" : \"arm64\"\n}" > ultra96.json
mv Ultra96.dcf ./pyxir/python/pyxir/contrib/target/components/DPUCZDX8G/
mv ultra96.json ./pyxir/python/pyxir/contrib/target/components/DPUCZDX8G/
cd pyxir
python3 setup.py install
