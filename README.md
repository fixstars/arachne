# WIP: arachne v0.2

```sh
arachne
├── 3rdparty
│   └── tvm
├── conda
├── docker
├── docs
├── pyproject.toml
├── python
│   └── arachne
│       ├── cli
│       ├── data.py
│       ├── pipeline.py
│       ├── runtime
│       ├── tools
│       └── util.py
├── scripts
└── tests
```


# How to setup a dev-container

1. First download the source code from Git.
2. Build and Run Docker Container.
3. Setup Python Virutalenv by Poetry.


## Expected Environment
* OS: Ubuntu 18.04
* CUDA: 10.2
* cuDNN: 8
* TensorRT: 7.2.3
* TensorFlow: 2.4.1
* PyTorch: 1.8.0
* ONNX and ONNX Runtime: 1.6


## Get Source from Git

```sh
git clone --recursive -b feature/v0.2 ssh://git@gitlab.fixstars.com:8022/arachne/arachne.git arachne
```

## Setup for Development Container

### Method 1: Visual Studio Code (Recommended)

If you use the Visual Studio Code (VSCode), you can use a development container via VSCode.
Please refer [the official document](https://code.visualstudio.com/docs/remote/containers>) to setup your VSCode.



## Create Python Virtual Environment

```sh
# create a virtual env to /workspaces/arachne/.venv
poetry install
poetry shell

# install other required packages
./scripts/build_tvm.sh
./scripts/install_torch2trt.sh
```

## Check Arachne Works Correctly

```sh
pytest tests --forked
```

Note that, `--forked` is required to make sure that the GPU memory is released for each test.

## Open document in local

```sh
sphinx-autobuild docs docs/_build --watch python/arachne
```