TVM
===

The details are described in :ref:`arachne.tools.tvm.TVMConfig <api-tools-tvm>`.

Using from CLI
--------------

.. code:: bash

    python -m arachne.tools.tvm \
        input=/path/to/model \
        input_spec=/path/to/model_spec.yaml \
        output=output.tar \
        tools.tvm.cpu_target=x86-64 \
        tools.tvm.cpu_attr=[+fma,+avx2] \
        tools.tvm.composite_target=[tensorrt,cpu]



Using in your code
------------------

.. code:: python

    from arachne.data import Model
    from aracune.utils import get_model_spec
    import arachne.tools.tvm

    model_file = "mobilenet.h5"
    input_model = Model(model_file, spec=get_model_spec(model_file))

    # Overwrite the spec for single-batch
    input_model.spec.inputs[0].shape = [1, 224, 224, 3]
    input_model.spec.outputs[0].shape = [1, 1000]

    # Setup tvm config
    cfg = arachne.tools.tvm.TVMConfig()
    cfg.cpu_target = "x86-64"
    cfg.cpu_attr = ['+fma', '+avx2']
    cfg.composite_target = ['tensorrt', 'cpu']

    output = arachne.tools.tvm.run(input=input, cfg=cfg)

