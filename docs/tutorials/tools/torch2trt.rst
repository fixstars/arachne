Torch2TRT
=========

The `torch2trt <https://github.com/NVIDIA-AI-IOT/torch2trt>`_ is a PyTorch to TensorRT converter.


Prepare a Model
---------------

First, we have to prepare a model to be used in this tutorial.
Here, we will use a pre-trained model of the ResNet-18 from `torchvision.models`.

.. code:: python

    import torch
    import torchvision

    resnet18 = torchvision.models.resnet18(pretrained=True)
    torch.save(resnet18, f="/tmp/resnet18.pt")


Run Torch2TRT from Arachne
--------------------------

Now, let's optimize the model with the torch2trt by Arachne.
To use the tool, we have to specify `+tools=torch2trt` to `arachne.driver.cli`.
Available options can be seen by adding `--help`.

.. code:: bash

    python -m arachne.driver.cli +tools=torch2trt --help


Optimize with FP32 precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we will start with the simplest case.
You can optimize a TF model with FP32 precision by the following command.
Note that, the Pytorch model does not include the information about tensor specification.
So, we need to pass the YAML file indicating the shape information.

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

    python -m arachne.driver.cli +tools=torch2trt model_file=/tmp/resnet18.pt model_spec_file=/tmp/resnet18.yaml output_path=/tmp/output.tar


Optimize with FP16 precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To optimize with FP16 precision, set `true` to the `tools.torch2trt.fp16_mode` option.

.. code:: bash

    python -m arachne.driver.cli +tools=torch2trt model_file=/tmp/resnet18.pt model_spec_file=/tmp/resnet18.yaml output_path=/tmp/output.tar tools.torch2trt.fp16_mode=true


Optimize with INT8 Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To convert with INT8 precision, we need calibrate or estimate the range of all floating-point tensors in the model.
We provide an interface to feed the dataset to be used in the calibration.
First, we have to prepare a NPY file that contains a list of `np.ndarray` which is a dataset used for calibration.
Here, we use a dummy dataset for explanation because the IMAGENET dataset requires manual setups for users.

.. code:: python

    import numpy as np
    datasets = []
    shape = [1, 3, 224, 224]
    dtype = "float32"
    for _ in range(100):
        datasets.append(np.random.rand(*shape).astype(np.dtype(dtype)))  # type: ignore

    np.save("/tmp/calib_dataset.npy", datasets)


Next, specify `true` to the `tools.torch2trt.int8_mode` option and pass the NPY file to the `tools.torch2trt.int8_calib_dataset`.


.. code:: bash

    python -m arachne.driver.cli +tools=torch2trt model_file=/tmp/resnet18.pt model_spec_file=/tmp/resnet18.yaml output_path=/tmp/output.tar \
        tools.torch2trt.int8_mode=true tools.torch2trt.int8_calib_dataset=/tmp/calib_dataset.npy


Run Torch2TRT from Arachne Python Interface
-------------------------------------------

The following code shows an example of using the tool from Arachne Python interface.
The details of the API are described in :ref:`arachne.tools.torch2trt <api-tools-torch2trt>`.

.. code:: python

    from arachne.data import ModelSpec, TensorSpec
    from arachne.utils.model_utils import init_from_file, save_model
    from arachne.tools.torch2trt import Torch2TRT, Torch2TRTConfig

    model_file_path = "/tmp/resnet18.pt"
    input = init_from_file(model_file_path)
    spec = ModelSpec(
        inputs=[TensorSpec(name="input0", shape=[1, 3, 224, 224], dtype="float32")],
        outputs=[TensorSpec(name="output0", shape=[1, 1000], dtype="float32")],
    )
    input.spec = spec

    cfg = Torch2TRTConfig()

    # cfg.fp16_mode = True

    output = Torch2TRT.run(input, cfg)

    save_model(model=output, output_path="/tmp/output.tar")

Jupyter Notebook Link
---------------------
You can see a notebook for this tutorial `here <https://github.com/fixstars/arachne/blob/main/examples/tools/run_torch2trt.ipynb>`_.