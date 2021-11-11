# Arachne Core Python API

A framework for compiling, evaluation and export DNN models for edge devices.

## Getting Started (on local machine)

First, we describe a quick-start for testing arachne APIs on your local machine with containers.

We tested the following steps in this environment.
* OS: Ubuntu 18.04 64-bit
* Container: Docker >= 19.03
* GPU: Installing NVIDIA GPU


### Step 1: Setup Develop Environment including Whole Dependencies

We provide our development environment including whole dependencies (both required and optional).

#### Option 1: Pull a Pre-built Image from GitLab
If you can access our GitLab container registry (i.e., gitlab.fixstars.com:5005/arachne/arachne:latest), you can use a pre-built docker image by pulling it.

```sh
$ docker pull gitlab.fixstars.com:5005/arachne/arachne:latest
```

#### Option 2: Build a Docker Image by Yourself
Or, you can build a docker image for arachne dev-container by yourself.

```sh
$ cd {path-to-arachne-repository}
$ docker build -t arachne:latest -f docker/Dockerfile docker
```

### Step 2: Enter the Dev Container

When the docker image is ready, you can enter the container by `docker/run.sh`. You can use the same user name at host machine.

```
$ ./docker/run.sh arachne:latest <container-name>

...

$ <your-username-at-host>@<hostname>:/workspace$
```

### Step 3: Build & Install TVM
After entering the container, first you should build and install  TVM libraries that arachne depends on.
You can accomplish this by running `docker/install/install_tvm.sh`.

```sh
$ ./docker/install/install_tvm.sh
...
Installing collected packages: tvm
  Running setup.py develop for tvm
Successfully installed tvm
```

### Step 4: Try the Example Code on Local Machine

Now, you are ready to run examples/local/*.py at the host PC.
Please refer `examples/local/README.md` for more details.


## Execute Inference on Remote Devices by RPC

By through TVM RPC execution, you can run dnn models on remote devices from the host PC.
Using TVM RPC execution, you needs several settings.

We tested the following steps for listed devices.
* Device:
    * NVIDIA Jetson Series
      * Jetpack >= 4.4 (i.e. CUDA==10.2)
    * Coral Dev Board
      * Mendel Linux == 5.2

For internal arachne developers, you can use a TVM rpc tracker that is continuously deployed by GitLab.
The rpc information is as follows:
* RPC Host: dgx-1.fixstars.com:9100
* RPC Keys:
  * jetson-nano
  * jetson-xavier-nx
  * jetson-tx2

### Step 1 (Host): Start RPC tracker in the host PC
```sh
python3 -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=8888 &
```

### Step 2 (Remote Device): Setup in the device
Clone the same version of the arachne source code to your device.
And, build and install TVM runtime.
We prepare some useful scripts for this process.
For example, if you want to install TVM runtime on jetson devices, `install_jetson.sh` is what you want.

```sh
./device/script/install/install_jetson.sh
```

After this, set environments and start a RPC server in the device.
```sh
ARACHNE_DEPLOY_RPC_TRACKER=<rpc-tracker-host>:8888 ./device/script/setup/setup_jetson_nano.sh
```

### Step 3 (Host): Confirm regitrations RPC server into the RPC tracker

Run the following command in the host.

```sh
python3 -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=8888
```

<details>
<summary>Output example</summary>

```
Tracker address 0.0.0.0:8888

Server List
----------------------------
server-address  key
----------------------------
${device-ip-address}:xxxx    server:${key-name}
----------------------------

Queue Status
----------------------------------------
key                total  free  pending
----------------------------------------
jetson-nano           1      1        0
----------------------------------------
```
</details>

### Step 4 (Host): Try the Example Code using RPC

Now, you are ready to run examples/rpc/*.py at the host PC.
Please refer `examples/rpc/README.md` for more details.


## Structure of this Project

### `python/`
This directory contains core arachne apis that will be exported as a pip package.
Test files for each api are also placed in this directory.

### `examples/`
To understand how to use arachne apis, we provide some example scripts in here.

### `docker/`

Note that, the arachne pip package iteself does not claim any dnn framework packages (such as tensorflow, torch, and so on) as requirements.
This is because what dnn frameworks are needed is depend on the users.
For example, a user who only use the tensorflow does not expect that a pytorch package will be installed by the arachne pip package.
So, we assume that users of arachne have the required packages pre-installed.

However, if you want a sandbox develop environment, you can use `docker/Dockerfile`.
This file is used to build a docker image including all possible libraries/packages to be used by arachne apis.

### `device/`

This directory contains install & setup scripts for using the tvm runtime and rpc environment at edge devices


## Arachne pip package
We provide arachne pip package for quick installation.

First, you need the access permission to our private package registry.
In short, you have to create a persornal access token in https://gitlab.fixstars.com.

### How to Install Packages

You can install arachne pip package by the following commands.

We provide the TVM source code as a sdist package to build the TVM library at the installation time.
This is because the TVM build config depends on the environment where arachne will be used.
For example, if you want to try CPU only mode, GPU features are not required to be built.

To accomplish this, we use three environment variables (i.e., `TVM_SOURCE_DIR`, `TVM_CMAKE_CONFIG` and `BUILD_TVM_RUNTIME_ONLY`) to control the install behavior for TVM libraries.
+ `TVM_SOURCE_DIR` allows you to specify the source directory to be built (e.g., `export TVM_SOURCE_DIR=/path/to/tvm`).
+ `TVM_CMAKE_CONFIG` represents a path to `config.cmake` that describes the compilation options.
Please refer [the default configuration file](https://github.com/apache/tvm/blob/main/cmake/config.cmake) to understand what can be customized.
+ When you specify `BUILD_TVM_RUNTIME_ONLY=1`, only the runtime library will be built.

```sh
$ export TVM_CMAKE_CONFIG=<path/to/config.cmake>
$ export BUILD_TVM_RUNTIME_ONLY=1 # otherwise the whole tvm libraries will be built.
$ pip install arachne --index-url https://__token__:<your_personal_token>@gitlab.fixstars.com/api/v4/projects/1757/packages/pypi/simple
$ pip show arachne
...
Location: </path/to/site-packages>
$ export TVM_LIBRARY_PATH=</path/to/site-packages> # to identify which tvm library should be used
```

### Install with editable mode

If you want to install the arachne package with editable mode (i.e., `pip install -e`), please follow the bellow steps.
Note that, in this case, you should install the tvm python package independently.

```sh
$ pwd
/path/to/arachne/python
$ python3 -m pip install -e .
$ python3 -m pip show arachne
Name: arachne
Version: 0.0.0.dev0
Summary: UNKNOWN
Home-page: UNKNOWN
Author:
Author-email:
License: UNKNOWN
Location: /workspace/python
Requires:
Required-by:
```