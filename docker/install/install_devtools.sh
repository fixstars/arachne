#!/bin/bash

set -euo pipefail

apt-get update && apt-get install -y --no-install-recommends \
  vim neovim emacs tmux zsh
