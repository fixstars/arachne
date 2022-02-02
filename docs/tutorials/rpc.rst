
RPC: Run Your Model on Remote devices
=====================================

`arachne.runtime.rpc` provides remote execution on a device using RPC (remote procedure call).

Edge Setup
----------
Install arachne and build tvm on your edge device.

.. attention:: TODO: prepare edge setup step

.. code:: shell

    git clone


Specify the runtime and port number, and start the RPC server on the edge device.

.. code:: shell

    python -m arachne.runtime.rpc.server --port 5051 --runtime tflite

Run model using RPC
-------------------

| The following example assumes that you are running the RPC server on localhost.
| You can get a runtime client with :code:`arachne.runtime.rpc.init` in the same way
| that you create a runtime module :code:`arachne.runtime.init`` for local execution.

.. code:: python

    import tempfile

    import numpy as np
    from tvm.contrib.download import download

    import arachne.runtime.rpc
    import arachne.tools.tvm
    from arachne.runtime.rpc import TVMRuntimeClient

    with tempfile.TemporaryDirectory() as tmp_dir:

        url = (
            "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/tvm_mobilenet.tar"
        )
        tvm_package_path = tmp_dir + "/tvm_mobilenet.tar"
        download(url, tvm_package_path)

        dummy_input = np.array(np.random.random_sample([1, 224, 224, 3]), dtype=np.float32)  # type: ignore

        client = arachne.runtime.rpc.init(package_tar=tvm_package_path, rpc_host="localhost", rpc_port=5051)
        assert isinstance(client, TVMRuntimeClient)
        client.set_input(0, dummy_input)
        client.run()
        rpc_output = client.get_output(0)
        client.finalize()

.. attention::
    Only one client can be connected to one Server at the same time.
    Using a client in the loop of a data loader running in multiprocess may cause gRPC communication to fail.