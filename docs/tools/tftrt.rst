TF-TRT
======
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
    import arachne.tools.tftrt

    # Setup an input model
    model_path = "saved_model"
    input_model = Model(model_path, spec=get_model_spec(model_path))

    # Run the TF-TRT
    cfg = arachne.tools.tftrt.TFTRTConfig()
    cfg.precision_mode = "FP16"
    output = arachne.tools.tftrt.run(input_model, cfg)