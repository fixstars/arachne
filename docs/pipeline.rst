
Pipeline: Run Different Tools in Sereies
========================================

Users can different tools in sereies by `arachne.pipeline`.


Using from CLI
--------------

Run `arachne.pipeline` with specifying the execution order of the tools by `pipeline` option.
The options for each tool can be specified as well as executing each tool.

.. code:: bash

    python -m arachne.pipeline \
        input=./tmp/model.h5 \
        output=./tmp/output.tar \
        pipeline=[tflite_converter,tvm] \
        tools.tflite_converter.ptq.method=none \
        tools.tvm.cpu_attr=[+fma,+avx2] \
        tools.tvm.composite_target=[tensorrt,cpu]


Using in your code
------------------


.. code:: python

    from arachne.data import Model
    from aracune.utils import get_model_spec
    import arachne.pipeline

    model_path = "mobilenet.h5"
    input = Model(path=model_path, spec=get_model_spec(model_path))

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