
Tool: Model Conversion and Compilation
======================================

This page provides an introduction to use `arachne.tools`, a variety of DNN model converters and a compiler.

YAML File for Input/Output Tensor Specification
-----------------------------------------------

Note that, some tools require the input and output tensor information of the input DNN models.
This is because, in some DNN model format, such information is not always deterministic.
We can pass such information to the tools by a YAML file like below.


.. code:: YAML

   # tf-keras-mobilenet-spec.yaml for tvm

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


Tool Output
-----------

all of the `arachne.tools` output a tar file.
The tar file contains a converted or compiled DNN model and a yaml file that describes the runtime dependency and the tensor information of the model.

.. code:: bash

  output.tar
  ├── env.yaml
  └── tvm_package_0.tar


.. code:: YAML

  dependencies:
  - tensorrt: 7.2.3-1+cuda10.2
  - pip:
    - tvm: 0.8.0
  model_spec:
    inputs:
    - dtype: float32
      name: input_1
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
  tvm_device: cpu


Available Tools
---------------

.. toctree::
  :maxdepth: 1

  tvm
  tflite_converter
  tftrt
  torch2onnx
  torch2trt
  openvino_mo
  openvino2tf

