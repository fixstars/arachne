TF-TRT
======

TensorFlow with TensorRT (TF-TRT) is a Tensorflow integration for optimizing Tensorflow models to execute them with TensorRT.
We also support the Post-training quantization like Tensorflow Lite Converter.
The details are described in :ref:`arachne.tools.tftrt <api-tools-tftrt>`.

Using from CLI
--------------

.. code:: bash

    python -m arachne.tools.tflite_converter \
        input=/path/to/model \
        output=output.tar \
        tools.tftrt.precision_mode="FP16"


Using from your code
----------------------

.. code:: python

    from arachne.data import Model
    from aracune.utils import get_model_spec
    from arachne.tools.tftrt import TFTRT, TFTRTConfig

    # Setup an input model
    model_path = "saved_model"
    input_model = Model(model_path, spec=get_model_spec(model_path))

    # Run the TF-TRT
    cfg = TFTRTConfig()
    cfg.precision_mode = "FP16"
    output = TFTRT.run(input_model, cfg)