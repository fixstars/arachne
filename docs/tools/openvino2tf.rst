OpenVINO2Tensorflow
===================
The details are described in :ref:`arachne.tools.openvino2tf <api-tools-openvino2tf>`.

Using from CLI
--------------

.. code:: bash

    python -m arachne.tools.openvino2tf \
        input=/path/to/model \
        input_spec=/path/to/model_spec.yaml \
        output=output.tar


Using from your code
----------------------

.. code:: python

    from arachne.data import Model, ModelSpec, TensorSpec
    import arachne.tools.openvino2tf

    # Setup an input model
    model_path = "resnet18.xml"  # or /path/to/openvino_mo_output_dir
    spec = ModelSpec(
        inputs=[TensorSpec(name="input0", shape=[1, 3, 224, 224], dtype="float32")],
        outputs=[TensorSpec(name="output0", shape=[1, 1000], dtype="float32")],
    )
    input_model = Model(path=model_path, spec=spec)

    # Run the openvino2tensorflow
    cfg = arachne.tools.openvino2tf.OpenVINO2TFConfig()
    output = arachne.tools.openvino2tf.run(input_model, cfg)