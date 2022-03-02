
RPC: Run Your Model on Remote devices
=====================================

:code:`arachne.runtime.rpc` provides remote execution on a device using RPC (remote procedure call).

Start RPC server
----------------

| Please refer to the :doc:`setup_device` for device environment setup.
| Start the rpc server using the created venv and arachne: :code:`./setup.sh <env_dirname> <runtime_name> <port>`.
| You can specify either :code:`tvm, tflite, onnx` to :code:`<runtime_name>`.

The following example shows the TVM runtime server running on JetPack 4.6 on port 5051.

.. code:: shell

    cd arachne/device
    ./setup.sh jp46 tvm 5051

Or, you can also start server as the following:

.. code:: shell

    cd arachne/device
    source jp46/.venv/bin/activate
    python -m arachne.runtime.rpc.server --port 5051 --runtime tvm

Run model using RPC
-------------------

| You can get a runtime client with :code:`arachne.runtime.rpc.init` in the same way that you create a runtime module :code:`arachne.runtime.init` for local execution.


.. code:: python

    client = arachne.runtime.rpc.init(
        model_file="resnet18.onnx",
        rpc_host="192.168.xx.xx",
        rpc_port=5051
    )
    assert isinstance(client, ONNXRuntimeClient)
    client.set_input(0, input_data)
    client.run()
    rpc_output = client.get_output(0)

:code:`tests/runtime/rpc/device/test_edge.py` is test script that the results of the local execution and the RPC execution are correct.
Before running test, start rpc server on the edge device with :code:`./setup.sh [env dirname] [tvm|tflite|onnx] 5051`

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
~~~~~~~~~~~~~~~~

.. code:: shell

    pytest tests/runtime/rpc/device/test_edge.py::test_tflite_runtime_rpc \
    --edgetest \
    --rpc_host 192.168.xx.xx \
    --rpc_port 5051

ONNX runtime test
~~~~~~~~~~~~~~~~

.. code:: shell

    pytest tests/runtime/rpc/device/test_edge.py::test_onnx_runtime_rpc \
    --edgetest \
    --rpc_host 192.168.xx.xx \
    --rpc_port 5051

.. attention::
    Only one client can be connected to one Server at the same time.
    Using a client in the loop of a data loader running in multiprocess may cause gRPC communication to fail.