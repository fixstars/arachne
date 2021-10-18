# Example Code to Understand How to Use Arachne APIs

You can run each script by

```sh
python3 <script-name>.py
```

## Requirements

You have to
* install all required packages and libraries ahead.
    * e.g., `tvm` and `tensorflow` packages should be installed to run `inference.py`


## `inference.py`

This script is the example code for compiling, benchmarking, running inference with RPC.

We describe some variables that will be modified by users.
* `RPC_HOST` and `RPC_KEY`
    * The RPC information