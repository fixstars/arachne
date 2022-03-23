
Run Multiple Tools in a Pipeline
================================

In this tutorial, we explain how to use Arachne pipeline functionality.
In Arachne, we leverge an open-source Python framework for constructing pipelines (i.e., `Kedro <https://kedro.readthedocs.io/en/stable/index.html#>`_).
First, we describe how to use this feature from Arachne CLI, and then the way to execute from Python interface.


Prepare a Model
---------------

For this tutorial, we will be working with the TFLite Converter and TVM for compling a Tensorflow (Keras) model by TVM after converting it into a TFLite model.
First, we prepare a Tensorflow model representing ResNet-50 v2 like the previous tutorails.


.. code:: python

    import tensorflow as tf

    model = tf.keras.applications.resnet_v2.ResNet50V2()
    model.summary()
    model.save("/tmp/resnet50-v2.h5")


Construct and Run a Pipeline by `arachne.driver.pipeline`
---------------------------------------------------------

.. code:: bash

    python -m arachne.driver.pipeline \
        input=/tmp/resnet50-v2.h5 \
        output=/tmp/output.tar \
        pipeline=[tflite_converter,tvm] \
        tools.tflite_converter.ptq.method=fp16 \
        tools.tvm.cpu_attr=[+fma,+avx2] \
        tools.tvm.composite_target=[tensorrt,cpu]


Here, we specify the two tools in a pipline (`pipeline=[tflite_converter,tvm]`).
To configure the behavior of each tool, we can control it by modifying `tools.tflite_converter` and `tools.tvm` options.
In this example, the TFLite Converter first converts the input model in FP16 mode and the TVM compile the converted model for the TensorRT target with allowing to execute the remaining graph on CPU.


Construct and Run a Pipeline by Python Interface
------------------------------------------------

If you want to use pipeline functionality with Python interfaces, please import the `arachne.driver.pipeline` module.
First, you should setup the `arachne.driver.pipeline.PipelineConfig` object which is a config class for pipeline.
To specify the tools in pipeline, you should pass a list of tool names to `PipelineConfig.pipeline`.
The `arachne.driver.pipeline.get_default_tool_configs` is used for retrieving the default configs for specified configs and saving the result to `PipelineConfig.tools`.
To modify the behavior of each tool, you can change the value under `PipelineConfig.tools`.
Last, `arachne.driver.pipeline.run` is used for executing the pipeline.

.. code:: python

    from arachne.data import Model
    from arachne.utils.model_utils import get_model_spec, save_model
    from arachne.driver.pipeline import PipelineConfig, get_default_tool_configs, run

    # Prepare an input model
    model_path = "/tmp/resnet50-v2.h5"
    input = Model(path=model_path, spec=get_model_spec(model_path))

    # Construct a pipeline
    cfg = PipelineConfig()
    cfg.pipeline = ['tflite_converter', 'tvm']
    cfg.tools = get_default_tool_configs(cfg.pipeline)

    # Setup tflite_converter config
    cfg.tools['tflite_converter'].ptq.method = "fp16"

    # Setup tvm config
    cfg.tools['tvm'].cpu_target = "x86-64"
    cfg.tools['tvm'].cpu_attr = ['+fma', '+avx2']
    cfg.tools['tvm'].composite_target = ['tensorrt', 'cpu']

    output = run(input, cfg)

    save_model(model=output, output_path="/tmp/output.tar", tvm_cfg=cfg.tools['tvm'])