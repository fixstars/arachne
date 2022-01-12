
Runtime: Run Your Model
=======================

To test converted or compiled models, arachne has the runtime that wraps original runtimes.
Currently, the arachne runtime supports the onnx, tflite, and tvm model.

.. code:: python

    import arachne.runtime
    import numpy as np

    # Run MobileNet by arachne.runtime

    # Init runtime by the tar files that arachne.tools output
    runtime_module = arachne.runtime.init(package_tar="package.tar")

    # or init runtime by specific model files and environment files
    # runtime_module = arachne.runtime.init(model_file="tvm_package.tar", env_file="env.yaml")

    # Set an input
    input_data = np.array(np.random.random_sample([1, 224, 224, 3]), dtype=np.float32)
    runtime_module.set_input(0, input_data)

    # Run an inference
    runtime_module.run()

    # Get a result
    out = runtime_module.get_output(0)
