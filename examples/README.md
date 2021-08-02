# Example Code to Understand How to Use Arachne APIs

You can run each script by

```sh
python3 run_tflite.py
```

## Requirements

You have to
* install all required packages and libraries ahead.
    * e.g., `tvm` and `tflite` packages should be installed to run `run_tflite.py`
* setup the rpc enviroment to use edge devices.

## `run_tflite.py`

This script is the example code for compiling, benchmarking, running inference for a tflite model.

We describe some variables that will be modified by users.
* `MODEL_URI`
    * An uri for model file.
* `INPUT_INFO` and `OUTPUT_INFO`
    * The tensor information of this model.
* `OUTPUT_DIR`
    * The directory where this script will output some files.
* `TARGET_DEVICE`
    * A compile target for the tvm compiler
* `RPC_HOST` and `RPC_KEY`
    * The RPC information

## `run_exported_pkg.py`

This script is used for running an exported package at localhost/remote devices.

Possibly modified variables are:
* `PACKAGE_PATH`
    * A path for an exported package.
* `RPC_HOST` and `RPC_KEY`
    * The RPC information

## `run_multiple_pipelines.py` (WIP)

This script describes how to use `make_pipeline_candidates()` for running multiple pipelines.

## FAQ

### Disconnect RPC Tracker/Server during Benchmarking

We faces this case when benchmarking large models at edge devices with enabling TensorRT.
This is because the TensorRT builder consumes much resources.
As a result, rpc connections are unexpectedly closed.

We are trying to find a generic solution for this problem.
But, as a work-around solution, we recommend to use `run_exported_pkg.py` at the device.
In other words, first, we prepare an exported package (like `exported.tar` in `run_tflite.py:L78`) for running at device.
Then, we manually send the package to the device.

At the device side, we can benchmark by the following commands:
```sh
# make sure that the shell executing the python script alive
$ tmux

# modify PACKAGE_PATH to specify </path/to/exported.tar>
$ python3 </path/to/arachne-mvp/examples/run_exported_pkg.py>
```
