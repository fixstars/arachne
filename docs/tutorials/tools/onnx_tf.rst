Run onnx-tf from Arachne
========================

The `onnx-tf <https://github.com/onnx/onnx-tensorflow>`_ is a ONNX to Tensorflow converter.


Prepare a Model
---------------

First, we have to prepare a model to be used in this tutorial.
Here, we will use a pre-trained model of the ResNet-18 from onnx models.

.. code:: bash

    wget https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet18-v1-7.onnx?raw=true -O resnet18.onnx


Run onnx-tf from Arachne
------------------------

Now, let's convert the model with the onnx-tf by Arachne.
To use the tool, we have to specify `+tools=onnx_tf` to `arachne.driver.cli`.
Available options can be seen by adding `--help`.

.. code:: bash

    python -m arachne.driver.cli +tools=onnx_tf --help


You can convert a model by the following command:

.. code:: bash

    python -m arachne.driver.cli +tools=onnx_tf input=./resnet18.onnx output=./output.tar


Run onnx-tf from Arachne Python Interface
-----------------------------------------

The following code shows an example of using the tool from Arachne Python interface.

.. code:: python

    from arachne.data import Model
    from arachne.utils.model_utils import save_model, get_model_spec
    from arachne.tools.onnx_tf import ONNXTf, ONNXTfConfig

    model_file_path = "./resnet18.onnx"
    input = Model(path=model_file_path, spec=get_model_spec(model_file_path))

    cfg = ONNXTfConfig()

    output = ONNXTf.run(input, cfg)

    save_model(model=output, output_path="./output.tar")