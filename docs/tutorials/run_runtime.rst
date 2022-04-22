
Runtime: Run Your Model
=======================

To test converted or compiled models, arachne has the runtime that wraps original runtimes.
Currently, the arachne runtime supports the onnx, tflite, and tvm model.

.. code:: python

    import arachne.runtime
    import numpy as np

    # Run MobileNet by arachne.runtime

    # Init runtime by runtime name and the tar files that arachne.tools output
    runtime_module = arachne.runtime.init(runtime="tvm", package_tar="package.tar")

    # or init runtime by specific model files and environment files
    # runtime_module = arachne.runtime.init(runtime="tvm", model_file="tvm_package.tar", env_file="env.yaml")

    # Set an input
    input_data = np.array(np.random.random_sample([1, 224, 224, 3]), dtype=np.float32)
    runtime_module.set_input(0, input_data)

    # Run an inference
    runtime_module.run()

    # Get a result
    out = runtime_module.get_output(0)


RPC: Run Your Model on Remote devices
=====================================

| :code:`arachne.runtime.rpc` provides remote execution on a device using RPC (remote procedure call).
| To perform remote execution via RPC, initialize RuntimeClient with :code:`arachne.runtime.rpc.init`.
| RuntimeClient has the same methods as RuntimeModule, and requests data i/o and model execution to the RuntimeServer running on the edge device.
| RuntimeServicer has :code:`arachne.runtime.RuntimeModule` instance internally and provides model execution services.

.. code:: python

    client = arachne.runtime.rpc.init(
        runtime="onnx",
        model_file="resnet18.onnx",
        rpc_host="192.168.xx.xx",
        rpc_port=5051
    )
    client.set_input(0, input_data)
    client.run()
    rpc_output = client.get_output(0)

RuntimeClient locks the server when initialization to block other clients from connecting to the RuntimeServer, and unlocks the server when the RuntimeClient instance is deleted or when the :code:`finalize` method is called.

.. attention::
    Only one client can be connected to one Server at the same time.
    Using a client in the loop of a data loader running in multiprocess may cause gRPC communication to fail.

Start RPC server
----------------

| Please refer to the :doc:`setup_device` for device environment setup.
| Start the rpc server using the created venv and arachne: :code:`./setup.sh <env_dirname> <port>`.

The following example shows the runtime server running on JetPack 4.6 on port 5051.

.. code:: shell

    cd arachne/device
    ./setup.sh jp46 5051

Or, you can also start server as the following:

.. code:: shell

    cd arachne/device
    source jp46/.venv/bin/activate
    python -m arachne.runtime.rpc.server --port 5051


Test
----

:code:`tests/runtime/rpc/device/test_edge.py` is test script that the results of the local execution and the RPC execution are correct.
Before running test, start rpc server on the edge device with :code:`./setup.sh [env dirname] 5051`

TVM runtime test
~~~~~~~~~~~~~~~~

| You must specify device name to :code:`--tvm_target_device` for tvm model compile.
| The device name is the name of the TVMConfig yaml file in the :code:`python/arachne/config/tvm_target` directory.

.. code:: shell

    pytest tests/runtime/rpc/device/test_edge.py::test_tvm_runtime_rpc \
    --edgetest \
    --tvm_target_device jetson-xavier-nx \
    --rpc_host 192.168.xx.xx \
    --rpc_port 5051

TfLite runtime test
~~~~~~~~~~~~~~~~~~~

.. code:: shell

    pytest tests/runtime/rpc/device/test_edge.py::test_tflite_runtime_rpc \
    --edgetest \
    --rpc_host 192.168.xx.xx \
    --rpc_port 5051

ONNX runtime test
~~~~~~~~~~~~~~~~~

.. code:: shell

    pytest tests/runtime/rpc/device/test_edge.py::test_onnx_runtime_rpc \
    --edgetest \
    --rpc_host 192.168.xx.xx \
    --rpc_port 5051


Jupyter Notebook Link
---------------------
You can see a notebook for this tutorial `here <https://github.com/fixstars/arachne/blob/main/examples/run_runtime.ipynb>`_.
