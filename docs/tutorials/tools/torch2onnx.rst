torch2onnx
==========

The torch2onnx  is a tool of Arachne that wraps `torch.onnx.export <https://pytorch.org/docs/stable/onnx.html>`_ .
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
    from arachne.tools.torch2onnx import Torch2ONNX, Torch2ONNXConfig

    # Setup an input model
    model_path = "resnet18.pt"
    spec = ModelSpec(
        inputs=[TensorSpec(name="input0", shape=[1, 3, 224, 224], dtype="float32")],
        outputs=[TensorSpec(name="output0", shape=[1, 1000], dtype="float32")],
    )
    input_model = Model(path=model_path, spec=spec)

    # Run the torch2onnx
    cfg = Torch2ONNXConfig()
    output = Torch2ONNX.run(input_model, cfg)