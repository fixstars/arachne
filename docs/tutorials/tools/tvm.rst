TVM
===

The `TVM <https://tvm.apache.org/>`_ is a deep learning compiler that enables users to compile their DNN models for various hardware targets.
In Arachne, we support it by providing interfaces that wrap `tvm.relay.build`.
The details of possible options are described in :ref:`arachne.tools.tvm.TVMConfig <api-tools-tvm>`.


Example: CLI
------------

To use the TVM, you have to pass `+tools=tvm` for `arachne.driver.cli` and then give input and output information and tool specific options (`tools.tvm.[options]`).

.. code:: bash

    python -m arachne.driver.cli \
        +tools=tvm
        input=/path/to/model \
        input_spec=/path/to/model_spec.yaml \
        output=output.tar \
        tools.tvm.cpu_target=x86-64 \
        tools.tvm.cpu_attr=[+fma,+avx2] \
        tools.tvm.composite_target=[tensorrt,cpu]


Example: Python Interface
-------------------------

For python interface, a module of the TVM is defined in `arachne.tools.tvm`.
`TVM` is a runner class and executes `tvm.relay.build` with taking an input model and a configuration (`TVMConfig`) that controls the build behavior.

.. code:: python

    from arachne.data import Model
    from aracune.utils import get_model_spec
    from arachne.tools.tvm import TVM, TVMConfig

    # Define an input model
    model_file = "mobilenet.h5"
    input_model = Model(model_file, spec=get_model_spec(model_file))

    # Overwrite the spec for single-batch
    input_model.spec.inputs[0].shape = [1, 224, 224, 3]
    input_model.spec.outputs[0].shape = [1, 1000]

    # Setup tvm config
    cfg = TVMConfig()
    cfg.cpu_target = "x86-64"
    cfg.cpu_attr = ['+fma', '+avx2']
    cfg.composite_target = ['tensorrt', 'cpu']

    output = TVM.run(input=input, cfg=cfg)


Pre-Defined TVM Configurations
------------------------------

For ease of setting the tvm-build configuration, we provide some pre-defined configurations.
To access the configurations, you have to add `+tvm_target=<target-name>` for CLI or use `arachne.tools.tvm.get_predefined_config` for python interface.

Available pre-defined config names are

* dgx-1
* dgx-s
* jetson-nano
* jetson-xavier-nx
* rasp4b64

.. code:: bash

    python -m arachne.tools.tvm \
        input=/path/to/model \
        input_spec=/path/to/model_spec.yaml \
        output=output.tar \
        +tvm_target=dgx-1


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
    cfg = arachne.tools.tvm.get_predefined_config("dgx-1")
    output = arachne.tools.tvm.run(input=input, cfg=cfg)