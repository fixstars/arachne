#!/bin/bash

set -euo pipefail

python3 -m pip install pip==21.0.1

python3 -m pip install \
    poetry six numpy decorator cython scipy tornado pytest pytest-xdist pytest-profiling \
    mypy orderedset attrs requests Pillow packaging cloudpickle synr ffi-navigator flake8 autopep8

python3 -m pip install pycocotools

python3 -m pip install --upgrade keyrings.alt 
