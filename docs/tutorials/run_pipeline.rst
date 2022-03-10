
Run Multiple Tools in a Pipeline
================================

To run multiple tools in sereies, we support pipeline functionality by leveraging an open-source Python framework for constructing pipelines (i.e., `Kedro <https://kedro.readthedocs.io/en/stable/index.html#>`_).
Here, we explain how to use the pipeline from CLI and Python interface.


Prepare a Test Model
--------------------

For this tutorial, we will be working with the TFLite Converter and TVM for compling a Tensorflow (Keras) model by TVM after converting it to a TFLite model.
First, we prepare a Tensorflow model representing ResNet-50 v2 like the previous tutorail.


.. code:: python

    model = tf.keras.applications.resnet_v2.ResNet50V2()
    model.save("resnet50-v2.h5")


Using Arachne CLI (i.e., `arachne.driver.pipeline`)
---------------------------------------------------

Now, you are ready to run a pipeline by using `arachne.driver.pipeline` which is a CLI for constructing and executing a pipeline including multiple tools.
To define a pipeline, you have to specify the `pipeline` option that takes a list of tool names.
To configure the tool behavior, you can use tool specific options as well as `arachne.driver.cli`.

.. code:: bash

    python -m arachne.pipeline \
        input=./tmp/model.h5 \
        output=./tmp/output.tar \
        pipeline=[tflite_converter,tvm] \
        tools.tflite_converter.ptq.method=none \
        tools.tvm.cpu_attr=[+fma,+avx2] \
        tools.tvm.composite_target=[tensorrt,cpu]


Using Arachne Python Interface
------------------------------

If you want to use pipeline functionality with Python interfaces, you can use the `arachne.pipeline` module.
`arachne.pipeline.PipelineConfig` is a config object.
`arachne.pipeline.get_default_tool_configs` is a function for retrieving default configurations of the specified tools.

.. code:: python

    from arachne.data import Model
    from aracune.utils import get_model_spec
    import arachne.pipeline

    # Prepare an input model
    model_path = "mobilenet.h5"
    input = Model(path=model_path, spec=get_model_spec(model_path))

    # Construct a pipeline
    cfg = arachne.pipeline.PipelineConfig()
    cfg.pipeline = ['tflite_converter', 'tvm']
    cfg.tools = arachne.pipeline.get_default_tool_configs(pipeline)

    # Setup tflite_converter config
    cfg.tools.tflite_converter.ptq.method = "none"

    # Setup tvm config
    cfg.tools.tvm.cpu_target = "x86-64"
    cfg.tools.tvm.cpu_attr = ['+fma', '+avx2']
    cfg.tools.tvm.composite_target = ['tensorrt', 'cpu']

    output = arachne.pipeline.run(input, cfg)