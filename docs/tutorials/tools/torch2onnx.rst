Torch2ONNX
==========

Here, we explain how to use the Torch2ONNX tool from Arachne.


Prepare a Model
---------------

First, we have to prepare a model to be used in this tutorial.
Here, we will use a pre-trained model of the ResNet-18 from `torchvision.models`.

.. code:: python

    import torch
    import torchvision

    resnet18 = torchvision.models.resnet18(pretrained=True)
    torch.save(resnet18, f="/tmp/resnet18.pt")


Run Torch2ONNX from Arachne
---------------------------

Now, let's convert the model into an ONNX model by Arachne.
To use the tool, we have to specify `+tools=torch2onnx` to `arachne.driver.cli`.
Available options can be seen by adding `--help`.

.. code:: bash

    python -m arachne.driver.cli +tools=torch2onnx --help


Passing the Pytorch model and it's tensor specification, the tool will covert the model into an ONNX model by the following command.

.. code:: bash

    cat /tmp/resnet18.yaml

    inputs:
    - dtype: float32
        name: input0
        shape:
        - 1
        - 3
        - 224
        - 224
    outputs:
    - dtype: float32
        name: output0
        shape:
        - 1
        - 1000



.. code:: bash

    python -m arachne.driver.cli +tools=torch2onn model_file=/tmp/resnet18.pt model_spec_file=/tmp/resnet18.yaml output_path=/tmp/output.tar


Run Torch2ONNX from Arachne Python Interface
--------------------------------------------

The following code shows an example of using the tool from Arachne Python interface.
The details are described in :ref:`arachne.tools.torch2onnx <api-tools-torch2onnx>`.

.. code:: python

    from arachne.data import ModelSpec, TensorSpec
    from arachne.utils.model_utils import init_from_file, save_model
    from arachne.tools.torch2onnx import Torch2ONNX, Torch2ONNXConfig

    model_file_path = "/tmp/resnet18.pt"
    input = init_from_file(model_file_path)
    spec = ModelSpec(
        inputs=[TensorSpec(name="input0", shape=[1, 3, 224, 224], dtype="float32")],
        outputs=[TensorSpec(name="output0", shape=[1, 1000], dtype="float32")],
    )
    input.spec = spec

    cfg = Torch2ONNXConfig()

    output = Torch2ONNX.run(input, cfg)

    save_model(model=output, output_path="/tmp/output.tar")

Jupyter Notebook Link
---------------------
You can see a notebook for this tutorial `here <https://github.com/fixstars/arachne/blob/main/examples/tools/run_torch2onnx.ipynb>`_.