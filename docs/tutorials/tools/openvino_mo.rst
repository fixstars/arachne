OpenVINO Model Optimizer
========================
The `OpenVINO Model Optimizer <https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html>`_ is a command line tool that converts the various kinds of DNN models to the OpenVINO Intermediate Representation (IR).
We expect to use this tool for an intermidiate step for converting a Pytorch/ONNX model (NCHW) to a Tensorflow model (NHWC) without incurring many transpose operators. An example of the conversion step is described at :ref:`tutorials/tools/openvino2tensorflow <tutorials_openvino2tf>`.

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