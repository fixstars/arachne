OpenVINO Model Optimizer
========================
The details are described in :ref:`arachne.tools.openvino_mo <api-tools-openvino-mo>`.

Using from CLI
--------------

.. code:: bash

    python -m arachne.tools.openvino_mo \
        input=/path/to/model \
        input_spec=/path/to/model_spec.yaml \
        output=output.tar


Using from your code
----------------------

.. code:: python

    from arachne.data import Model, ModelSpec, TensorSpec
    from aracune.utils import get_model_spec
    import arachne.tools.openvino_mo

    # Setup an input model
    model_path = "resnet18.onnx"
    input_model = Model(path=model_path, spec=get_model_spec(model_path))

    # Run the openvino model optimizer
    cfg = arachne.tools.openvino_mo.OpenVINOModelOptConfig()
    output = arachne.tools.openvino_mo.run(input_model, cfg)