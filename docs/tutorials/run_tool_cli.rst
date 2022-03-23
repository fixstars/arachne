
Run a Tool from Arachne CLI
===========================

In this tutorial, we explain how to use Arachne CLI (i.e., `arachne.driver.cli`) for running a tool in Arachne.
Here, we will be working with a tool of TVM to compile ResNet-50 v2 from the Tensorflow Keras Applications.
TVM is a deep learning compiler with supporting various DNN models as its input.
ResNet-50 v2 is one of the famous convolutional neural networks to classify images.

Prepare a Model
---------------

First, we prepare an input model by using a Tensorflow Keras API.


.. code:: python


  import tensorflow as tf

  model = tf.keras.applications.resnet_v2.ResNet50V2()
  model.summary()
  model.save("/tmp/resnet50-v2.h5")


Apply a Tool to the Input Model
-------------------------------

Next, let's try to execute the TVM taking the prepared model as it's input from Arachne CLI.
Typically, you can start with the following command.

.. code:: bash

  python -m arachne.driver.cli +tools=tvm input=/tmp/resnet50-v2.h5 output=/tmp/output.tar


Deals with the Dynamic Shape
----------------------------

Now you can see there is a something worng because the TVM cannot deal with the negative shape value (, or the dynamic shape).
TVM requires to specify the static shape for the networks that have dynamic shapes.
To address this problem, we provide an option (i.e., `input_spec`) to specify the tensor specification of the input model.
Users can pass a path to the YAML file that describes such information.
For example, the file for this case looks like below.

.. code:: YAML

  # /tmp/resnet50-v2.yaml
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


Finally, you can compile it.

.. code:: bash

  python -m arachne.driver.cli +tools=tvm input=/tmp/resnet50-v2.h5 output=/tmp/output.tar input_spec=/tmp/resnet50-v2.yaml


Try Tool-Specific Configurations
--------------------------------

You can configure the tool behavior by passing specific values to options.
To understand what options are available, you just add `--help` to the previous command.

.. code:: bash

  python -m arachne.driver.cli +tools=tvm input=/tmp/resnet50-v2.h5 output=/tmp/output.tar input_spec=/tmp/resnet50-v2.yaml --help


Here, we only explain a simple usage to compile for TensorRT and CUDA targets for space problem. Please refer to the API documentation for `arachne.tools` to know details.
To compile for TensorRT and CUDA targets, you should set `tools.tvm.***` options appropriately like below:

.. code:: bash

  python -m arachne.driver.cli +tools=tvm input=/tmp/resnet50-v2.h5 output=/tmp/output.tar input_spec=/tmp/resnet50-v2.yaml tools.tvm.composite_target=[tensorrt,cuda]


Pre-defined Configs for TVM Target
----------------------------------

To ease setup the TVM target, we provide pre-defined configurations for some devices.
For example, you can pass `+tvm_target=dgx-1` for Nvidia DGX-1 instead of specifying multiple options.


.. code:: bash

  python -m arachne.driver.cli +tools=tvm input=/tmp/resnet50-v2.h5 output=/tmp/output.tar input_spec=/tmp/resnet50-v2.yaml +tvm_target=dgx-1


Check Output TAR File
---------------------

All of the Arachne Tool outputs a TAR file.
The file contains a converted or compiled DNN model and a YAML file that describes the runtime dependency and the tensor information of the model.

.. code:: bash

  output.tar
  ├── env.yaml
  └── tvm_package_0.tar

.. code:: bash

  # env.yaml
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
  tvm_device: cuda