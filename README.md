# Arachne

A framework for compiling, evaluation and export DNN models for edge devices.

## Dependencies
* Host
    * Ubuntu 18.04 64-bit
    * Docker >= 19.03
    * Installing NVIDIA GPU
* Device:
    * NVIDIA Jetson Series
      * Jetpack >= 4.4 (i.e. CUDA==10.2)
    * Coral Dev Board
      * Mendel Linux == 5.2

## Installation
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

## Getting Started
In this tutorial, we try to compile SSD MobileNet V1 COCO pretrained model and benchmark on RaspberryPi 4.


### Compile a model for the device
Compile the trained model.

Please reference [compile test codes](https://gitlab.fixstars.com/arachne/arachne-mvp/-/blob/master/python/tests/compile_test.py).

TODO: More details

### Setup for benchmark performances of the model on the device
By through TVM RPC execution, you can run benchmarks on remote devices from the host PC.
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
Confirm the registration RPC server into the RPC tracker.

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

### Benchmark performances of the model on the device
Benchmark performances of the compiled model.

Please reference [benchmark test codes](https://gitlab.fixstars.com/arachne/arachne-mvp/-/blob/master/python/tests/benchmark_test.py).

TODO: More details



```sh
python3 python/arachne/benchmark.py --package=raspi4.tar.gz --device=raspi4 --rpc-tracker=localhost:8888 --rpc-key=raspi4
```

<details>
<summary>Output example running on Jetson Xavier NX</summary>

```
Node Name                            Ops                                  Time(us)  Time(%)  Shape          Inputs  Outputs
---------                            ---                                  --------  -------  -----          ------  -------
tensorrt_0                           tensorrt_0                           7239.27   77.228   (1, 91, 1917)  1       2
tensorrt_0                           tensorrt_0                           7239.27   77.228   (1, 1917, 4)   1       2
fused_vision_multibox_transform_loc  fused_vision_multibox_transform_loc  946.83    10.101   (1, 1917, 6)   3       2
fused_vision_multibox_transform_loc  fused_vision_multibox_transform_loc  946.83    10.101   (1,)           3       2
tensorrt_97                          tensorrt_97                          318.178   3.394    (1, 10, 4)     4       1
tensorrt_95                          tensorrt_95                          279.617   2.983    (1, 7668)      4       1
tensorrt_96                          tensorrt_96                          149.562   1.596    (1, 10, 6)     1       1
tensorrt_98                          tensorrt_98                          120.618   1.287    (1, 10)        1       1
tensorrt_99                          tensorrt_99                          108.717   1.16     (1, 10)        1       1
fused_vision_non_max_suppression     fused_vision_non_max_suppression     87.546    0.934    (1, 1917, 6)   4       1
fused_split_2                        fused_split_2                        50.272    0.536    (1, 1917, 1)   2       4
fused_split_2                        fused_split_2                        50.272    0.536    (1, 1917, 1)   2       4
fused_split_2                        fused_split_2                        50.272    0.536    (1, 1917, 1)   2       4
fused_split_2                        fused_split_2                        50.272    0.536    (1, 1917, 1)   2       4
fused_vision_get_valid_counts        fused_vision_get_valid_counts        38.288    0.408    (1,)           2       3
fused_vision_get_valid_counts        fused_vision_get_valid_counts        38.288    0.408    (1, 1917, 6)   2       3
fused_vision_get_valid_counts        fused_vision_get_valid_counts        38.288    0.408    (1, 1917)      2       3
fused_split_1                        fused_split_1                        35.04     0.374    (1, 10, 1)     1       6
fused_split_1                        fused_split_1                        35.04     0.374    (1, 10, 1)     1       6
fused_split_1                        fused_split_1                        35.04     0.374    (1, 10, 1)     1       6
fused_split_1                        fused_split_1                        35.04     0.374    (1, 10, 1)     1       6
fused_split_1                        fused_split_1                        35.04     0.374    (1, 10, 1)     1       6
fused_split_1                        fused_split_1                        35.04     0.374    (1, 10, 1)     1       6
Total_time                           -                                    9373.938  -        -              -       -
Execution time summary:
 mean (s)   max (s)    min (s)    std (s)
 0.00881    0.00918    0.00864    0.00016
```
</details>


## Arachne pip package

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
