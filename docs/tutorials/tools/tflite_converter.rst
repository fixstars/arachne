Tensorflow Lite Converter
=========================
The details are described in :ref:`arachne.tools.tflite_converter <api-tools-tflite-converter>`.

Using from CLI
--------------

.. code:: bash

    python -m arachne.tools.tflite_converter \
        input=/path/to/model \
        output=output.tar \
        tools.tflite_converter="dynamic_range"


Using from your code
----------------------

.. code:: python

    from arachne.data import Model
    from aracune.utils import get_model_spec
    import arachne.tools.tflite_converter

    # Setup an input model
    model_file = "mobilenet.h5"
    input_model = Model(model_file, spec=get_model_spec(model_file))

    # Run the tflite converter
    cfg = arachne.tools.tflite_converter.TFLiteConverterConfig()
    cfg.ptq.method = "dynamic_range"
    output = arachne.tools.tflite_converter.run(input_model, cfg)


Use representative datasets for PTQ
-----------------------------------

For INT8 PTQ, we need representative datasets to calibration.
We support passing the datasets to tflite_converter via `*.npy` files that contains `List[np.ndarray]`.


.. code:: python

    # Example of creating the dataset from imangenet 2012 in tfds

    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tfds_dir = "/path/to/tfds_dir"
    data, _ = tfds.load(
        name="imagenet2012",
        split=["validation"],
        data_dir=tfds_dir,
        with_info=True,
    )

    ds = data[0].batch(1).prefetch(tf.data.AUTOTUNE)

    res = []
    data_num = 100
    for data in ds.take(data_num).as_numpy_iterator():
        image = tf.image.resize(data["image"], [224, 224])
        image = tf.keras.applications.imagenet_utils.preprocess_input(
            x=image, mode='tf'
        )
        res.append(image.numpy())
    np.save("imagenet2012.npy", res)