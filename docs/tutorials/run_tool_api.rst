
Run a Tool from Arachne Python Interface
========================================

Here, we explain how to use Arachne Python interface for running a tool.
Like the example for Arachne CLI, we will be working with a tool of TVM to compile ResNet-50 v2 from the Tensorflow Keras Applications.

Prepare a Test Model
--------------------

First, we prepare an input model by using a Tensorflow Keras API.


.. code:: python

  import tensorflow as tf

  model = tf.keras.applications.resnet_v2.ResNet50V2()
  model.summary()
  model.save("/tmp/resnet50-v2.h5")



Initialize `arachne.data.Model`
-------------------------------


In Arachne, DNN models are managed by `arachne.data.Model` data objects which keeps the model file path and the tensor specification of the model.
Each tool takes the object as its input and outputs another `Model` as a result.
`arachne.utils.model_utils.init_from_file` is a helper function to initialize a `Model` instance from a model file.

.. code:: python

  from arachne.utils.model_utils import init_from_file


  model_file_path = "/tmp/resnet50-v2.h5"
  input = init_from_file(model_file_path)

  print(input)


Deal with the Dynamic Shape
---------------------------

According to the example of the Arachne CLI, we have to deal with the dynamic shape (i.e., `-1` in the shape).
Users can modify the tensor specification directly by updating the value.

.. code:: python

  input.spec.inputs[0].shape = [1, 224, 224, 3]
  input.spec.outputs[0].shape = [1, 1000]
  print(input)


Execute a Tool from Arachne Python Interface
--------------------------------------------

The module for each tool is defined under `arachne.tools`.
In this example, the module for the TVM is implemented at `arachne.tools.tvm`.
There are two main classes.
First, `TVMConfig` is a class for configuraing the behavior of the TVM compile processing.
Next, `TVM` is a wrapper class for executing `tvm.relay.build`.
To run the TVM, users will call a static method (i.e, `TVM.run`) with passing the input model and the config object.


.. code:: python

  from arachne.tools.tvm import TVMConfig, TVM

  tvm_cfg = TVMConfig()
  output = TVM.run(input=input, cfg=tvm_cfg)


Save the Result
---------------

To save the result as a TAR file, `arachne.utils.model_utils.save_model` should be called.

.. code:: python

  from arachne.utils.model_utils import save_model

  save_model(model=output, output_path="/tmp/output.tar", tvm_cfg=tvm_cfg)


Pre-defined Configs for TVM Target
-----------------------------------

To use pre-defined configs for some TVM target, you can use `arachne.tools.tvm.get_predefined_config`.

.. code:: python

  from arachne.tools.tvm import get_predefined_config

  conf = get_predefined_config("dgx-1")
  print(conf)


Jupyter Notebook Link
---------------------
You can see a notebook for this tutorial `here <https://github.com/fixstars/arachne/blob/main/examples/run_api.ipynb>`_.