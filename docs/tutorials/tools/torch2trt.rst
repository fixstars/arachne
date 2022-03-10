torch2trt
=========
The `torch2trt <https://github.com/NVIDIA-AI-IOT/torch2trt>`_ is a PyTorch to TensorRT converter.
In Arachne, we support this tool by providing interfaces that wrap `torch2trt`.
The details are described in :ref:`arachne.tools.torch2trt <api-tools-torch2trt>`.

Using from CLI
--------------

.. code:: bash

    python -m arachne.tools.torch2trt \
        input=/path/to/model.pt \
        input_spec=/path/to/model.yaml \
        output=output.tar \
        toosl.torch2trt.precision = "FP16"


Using from your code
----------------------

.. code:: python

    from arachne.data import Model
    from aracune.utils import get_model_spec
    from arachne.tools.torch2trt import Torch2TRT, Torch2TRTConfig

    # Setup an input model
    model_path = "resnet18.pt"
    spec = ModelSpec(
        inputs=[TensorSpec(name="input0", shape=[1, 3, 224, 224], dtype="float32")],
        outputs=[TensorSpec(name="output0", shape=[1, 1000], dtype="float32")],
    )
    input_model = Model(path=model_path, spec=spec)

    # Run the torch2trt
    cfg = Torch2TRTConfig()
    cfg.precision = "FP16"
    output = Torch2TRT.run(input_model, cfg)