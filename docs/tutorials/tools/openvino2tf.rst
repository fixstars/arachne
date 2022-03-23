.. _tutorials_openvino2tf:

OpenVINO2Tensorflow
===================

The `openvino2tensorflow <https://github.com/PINTO0309/openvino2tensorflow>`_ is a script that converts the ONNX/OpenVINO IR model to Tensorflow model.

When you convert onnx model to tensorflow model by `onnx-tf`, the converted model includes many unnecessary transpose layers. This is because onnx has NCHW layer format while tensorflow has NHWC.
The inclusion of many unnecessary transpose layers causes performance degradation in inference.

By using openvino2tensorflow, you can avoid the inclusion of unnecessary transpose layers when converting a model from to tensorflow.
In this tutorial, we compare two convert methods and their converted models:

1. PyTorch -> (torch2onnx) -> ONNX -> (onnx-simplifier) -> ONNX -> (onnx-tf) -> Tensorflow -> (tflite_converter) -> TfLite
2. PyTorch -> (torch2onnx) -> ONNX -> (onnx-simplifier) -> ONNX -> (openvino_mo) -> OpenVino -> (openvino2tensorflow) -> Tensorflow -> (tflite_converter) -> TfLite

The developers of openvino2tensorflow provides the detail article about the advantage using openvino2tensorflow: `Converting PyTorch, ONNX, Caffe, and OpenVINO (NCHW) models to Tensorflow / TensorflowLite (NHWC) in a snap <https://qiita.com/PINTO/items/ed06e03eb5c007c2e102>`_


Create Simple Model
-------------------
Here we create and save a very simple PyTorch model to be converted.

.. code::  python

    import torch
    from torch import nn
    import torch.onnx

    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.Conv2d(16, 16, 3, padding=1),
    )
    torch.save(model.eval(), "./sample.pth")

Save model input and output information as yaml format for arachne.

.. code:: python

    yml = """
    inputs:
    - dtype: float32
        name: input
        shape:
        - 1
        - 3
        - 224
        - 224
    outputs:
    - dtype: float32
        name: output
        shape:
        - 1
        - 16
        - 224
        - 224
    """
    open("sample.yml", "w").write(yml)

Convert using onnx-tf
---------------------

You can apply multiple tools in sequence with :code:`arachne.pipeline`.
Models are converted in the following order:
PyTorch -> (torch2onnx) -> ONNX -> (onnx-simplifier) -> ONNX -> (onnx-tf) -> Tensorflow -> (tflite_converter) -> TfLite

.. code:: bash

    python -m arachne.driver.pipeline \
    +pipeline=[torch2onnx,onnx_simplifier,onnx_tf,tflite_converter] \
    input=./sample.pth \
    output=./pipeline1.tar \
    input_spec=./sample.yml

Extract tarfile and see network structure of the converted tflite model.
You can visualize model structure in netron: :code:`netron ./pipeline1/model_0.tflite`.

.. code:: bash

    mkdir -p pipeline1 && tar xvf pipeline1.tar -C ./pipeline1

.. code:: python

    import tensorflow as tf

    def list_layers(model_path):
        interpreter = tf.lite.Interpreter(model_path)
        layer_details = interpreter.get_tensor_details()
        interpreter.allocate_tensors()

        for layer in layer_details:
            print("Layer Name: {}".format(layer['name']))

    list_layers("./pipeline1/model_0.tflite")

.. code::

    Layer Name: serving_default_input.1:0
    Layer Name: transpose_2/perm
    Layer Name: transpose_1/perm
    Layer Name: Const
    Layer Name: convolution
    Layer Name: convolution_1
    Layer Name: Add;convolution_1;convolution;Const_1
    Layer Name: Add_1;convolution_1;Const_3
    Layer Name: Pad
    Layer Name: transpose_1
    Layer Name: Add;convolution_1;convolution;Const_11
    Layer Name: transpose_2
    Layer Name: Pad_1
    Layer Name: transpose_4
    Layer Name: Add_1;convolution_1;Const_31
    Layer Name: PartitionedCall:0

We have confirmed that the transpose layer is unexpectedly included.

Convert using openvino2tensorflow
---------------------------------
Next, try the second conversion method using openvino2tensorflow.
Models are converted in the following order:
PyTorch -> (torch2onnx) -> ONNX -> (onnx-simplifier) -> ONNX -> (openvino_mo) -> OpenVino -> (openvino2tensorflow) -> Tensorflow -> (tflite_converter) -> TfLite

.. code:: bash

    python -m arachne.driver.pipeline \
    +pipeline=[torch2onnx,onnx_simplifier,openvino_mo,openvino2tf,tflite_converter] \
    input=./sample.pth \
    output=./pipeline2.tar \
    input_spec=./sample.yml

Extract tarfile and see network structure of the converted tflite model.
You can visualize model structure in netron: :code:`netron ./pipeline2/model_0.tflite`.

.. code:: bash

    mkdir -p pipeline2 && tar xvf pipeline2.tar -C ./pipeline2

.. code:: python

    list_layers("./pipeline2/model_0.tflite")

.. code::

    Layer Name: serving_default_input_1:0
    Layer Name: model/zero_padding2d/Pad/paddings
    Layer Name: model/conv2d/Conv2D
    Layer Name: model/conv2d_1/Conv2D
    Layer Name: model/tf.math.add/Add;model/conv2d_1/Conv2D;model/conv2d/Conv2D;model/tf.math.add/Add/y
    Layer Name: model/tf.math.add_1/Add;model/conv2d_1/Conv2D;model/tf.math.add_1/Add/y
    Layer Name: model/zero_padding2d/Pad
    Layer Name: model/tf.math.add/Add;model/conv2d_1/Conv2D;model/conv2d/Conv2D;model/tf.math.add/Add/y1
    Layer Name: model/zero_padding2d_1/Pad
    Layer Name: StatefulPartitionedCall:0

We have confirmed that the transpose layer is NOT included.