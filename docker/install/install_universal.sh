#!/bin/bash

set -euo pipefail

git clone https://github.com/stillwater-sc/universal.git /opt/universal

# Use specific versioning tag.
(cd /opt/universal && git checkout e32899d551b53d758865fabd5fdd69eed35bfb0f)
