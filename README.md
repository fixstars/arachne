# Arachne Core Python API

A framework for compiling, evaluation and export DNN models for edge devices.

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

## Using Arachne Apis in Docker Environment

### Dependencies
* Host
    * Ubuntu 18.04 64-bit
    * Docker >= 19.03
    * Installing NVIDIA GPU
* Device:
    * NVIDIA Jetson Series
      * Jetpack >= 4.4 (i.e. CUDA==10.2)
    * Coral Dev Board
      * Mendel Linux == 5.2

### Step 1: Docker Build & Install tvm
```sh
cd {path-to-arachne-repository}
docker build -t arachne:latest -f docker/Dockerfile docker
```

After success the docker build, you can enter the container by the following command.

```sh
./docker/run.sh arachne:latest
export PYTHONPATH=/workspace/python:$PYTHONPATH
./docker/install/install_tvm.sh
```

After this, check installing tvm
```
python3 -c "import tvm"
```

Now, you are ready to run `examples/*.py` at the host PC.

### Step 2: Setup RPC Environment (optional)

By through TVM RPC execution, you can run dnn models on remote devices from the host PC.
Using TVM RPC execution, you needs several settings.

#### Start RPC tracker in the host PC
```sh
python3 -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=8888 &
```

#### Setup in the device
Build and install TVM runtime.
```sh
./device/script/install/install_jetson.sh
```

After this, set environments and start a RPC server in the device.
```sh
source ./device/script/setup/setup_jetson_nano.sh
```

#### Confirm regitrations RPC server into the RPC tracker

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

## Arachne pip package (WIP)
NOTE: this project is under construct, so there is no a stable pip package yet.
We recommend use arachne apis in the dockerized environment.

First, you have to create a persornal access token in https://gitlab.fixstars.com.
Then, you can install/publish a python pkg by the following commands

### Pip install

```
$ pip install arachne --index-url https://__token__:<your_personal_token>@gitlab.fixstars.com/api/v4/projects/1757/packages/pypi/simple
```

### Publish a new package

```
$ poetry publish --repository arachne -u <your_user_name> -p <your_access_token>
```
