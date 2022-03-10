
Run a Tool with Arachne CLI & Python Interface
==============================================

Here, we will explain how to execute an Arachne Tool.


Get a Test Model
----------------

For this tutorial, we will be working with a tool of TVM to compile ResNet-50 v2.
TVM is a deep learning compiler with supporting various DNN models as its input.
ResNet-50 v2 is one of the famous convolutional neural networks to classify images.


.. code:: bash

    wget https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx



Using Arachne CLI (i.e., `arachne.driver.cli`)
----------------------------------------------

Once we’ve downloaded the test model, the next step is to compile it by TVM through Arachne.
To accomplish that, we can use `arachne.driver.cli`, the Arachne command line driver to run our tool.

.. code:: bash

    python -m arachne.driver.cli \
        +tools=tvm
        input=resnet50-v2-7.onnx \
        output=output.tar


Using Arachne Python Interface
------------------------------

Or we can use the Arachne python interfance.

.. code:: python

    from arachne.data import Model
    from arachne.tools.tvm import TVMConfig, TVM
    from arachne.utils.model_utils import get_model_spec, save_model

    # Init a tool input
    model_file = "resnet50-v2-7.onnx"
    input_model = Model(model_file, spec=get_model_spec(model_file))

    # Run a tool
    tvm_cfg = TVMConfig()
    output = TVM.run(input=input_model, cfg=tvm_cfg)

    # Save the result
    save_model(model=output_model, output_path="output.tar", tvm_cfg=tvm_cfg)



Ouput TAR file
--------------

All of the Arachne Tool output is a TAR file.
The file contains a converted or compiled DNN model and a YAML file that describes the runtime dependency and the tensor information of the model.

.. code:: bash

  output.tar
  ├── env.yaml
  └── tvm_package_0.tar


.. code:: YAML

  dependencies:
  - pip:
    - tvm: 0.8.0
  model_spec:
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
  tvm_device: cpu

YAML File for Input/Output Tensor Specification
-----------------------------------------------

Note that, some tools require the input and output tensor information of the input DNN models.
This is because, in some DNN model format, such information is not always deterministic.
We can pass such information to the tools by using the `input_spec` option for CLI or by overwrite the model information for python interface.

For example, a typical usage for CLI are shown in:

.. code:: bash

    python -m arachne.driver.cli \
        +tools=tvm
        input=resnet50-v2-7.onnx \
        input_spec=resnet50-v2-7.yaml \
        output=output.tar

.. code:: YAML

   # resnet50-v2-7.yaml

   inputs:
   - dtype: float32
     name: input
     shape:
     - 1
     - 224
     - 224
     - 3
   outputs:
   - dtype: float32
     name: Identity
     shape:
     - 1
     - 1000


Or, for python interface, we can modify `Model.spec` attributes like below.

.. code:: python

    from arachne.data import Model
    from arachne.tools.tvm import TVMConfig, TVM
    from arachne.utils.model_utils import get_model_spec, save_model

    # Init a tool input
    model_file = "resnet50-v2-7.onnx"
    input_model = Model(model_file, spec=get_model_spec(model_file))

    # Overwrite the spec for single-batch
    input_model.spec.inputs[0].shape = [1, 3, 224, 224]
    input_model.spec.outputs[0].shape = [1, 1000]

    # Run a tool
    tvm_cfg = TVMConfig()
    output = TVM.run(input=input_model, cfg=tvm_cfg)

    # Save the result
    save_model(model=output_model, output_path="output.tar", tvm_cfg=tvm_cfg)


Available Tools
---------------

For detailed information of each tools, please refer to the following pages.

.. toctree::
  :maxdepth: 1

  tools/tvm
  tools/tflite_converter
  tools/tftrt
  tools/torch2onnx
  tools/torch2trt
  tools/openvino_mo
  tools/openvino2tf