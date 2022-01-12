torch2onnx
==========
The details are described in :ref:`arachne.tools.torch2onnx <api-tools-torch2onnx>`.

Using from CLI
--------------

.. code:: bash

    python -m arachne.tools.tflite_converter \
        input=/path/to/model \
        input_spec=/path/to/model.yaml \
        output=output.tar


Using from your code
----------------------

.. code:: python

    from arachne.data import Model
    from aracune.utils import get_model_spec
    import arachne.tools.torch2onnx

    # Setup an input model
    model_path = "resnet18.pt"
    spec = ModelSpec(
        inputs=[TensorSpec(name="input0", shape=[1, 3, 224, 224], dtype="float32")],
        outputs=[TensorSpec(name="output0", shape=[1, 1000], dtype="float32")],
    )
    input_model = Model(path=model_path, spec=spec)

    # Run the torch2onnx
    cfg = arachne.tools.torch2onnx.Torch2ONNXConfig()
    output = arachne.tools.torch2onnx.run(input_model, cfg)