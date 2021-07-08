#!/usr/bin/env bash

set -euo pipefail

if [ $# -ne 0 ]; then
    echo "Usage: /ci/test.sh"
    exit 1
fi

poetry run pytest python
