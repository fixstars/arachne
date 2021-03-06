Tensorflow Lite Converter
=========================

The `TensorFlow Lite Converter <https://www.tensorflow.org/lite/convert>`_ takes a TensorFlow model and generates a TensorFlow Lite model.
Here, we explain how to use the TFLite Converter from Arachne especially focusing on controlling the tool behavior.



Prepare a Model
---------------

First, we have to prepare a model to be used in this tutorial.
Here, we will use a ResNet-50 v2 model tuning for the `tf_flowers` dataset.

.. code:: python

    import tensorflow as tf
    import tensorflow_datasets as tfds

    # Initialize a model
    model = tf.keras.applications.resnet_v2.ResNet50V2(weights=None, classes=5)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    model.summary()

    # Load the tf_flowers dataset
    train_dataset, val_dataset = tfds.load(
        "tf_flowers", split=["train[:90%]", "train[90%:]"], as_supervised=True
    )

    # Preprocess the datasets
    def preprocess_dataset(is_training=True):
        def _pp(image, label):
            if is_training:
                image = tf.image.resize(image, (280, 280))
                image = tf.image.random_crop(image, (224, 224, 3))
                image = tf.image.random_flip_left_right(image)
            else:
                image = tf.image.resize(image, (224, 224))
            image = tf.keras.applications.imagenet_utils.preprocess_input(x=image, mode='tf')
            label = tf.one_hot(label, depth=5)
            return image, label
        return _pp


    def prepare_dataset(dataset, is_training=True):
        dataset = dataset.map(preprocess_dataset(is_training), num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.batch(16).prefetch(tf.data.AUTOTUNE)

    train_dataset = prepare_dataset(train_dataset, True)
    val_dataset = prepare_dataset(val_dataset, False)

    # Training
    model.fit(train_dataset, validation_data=val_dataset, epochs=20)

    model.evaluate(val_dataset)

    model.save("/tmp/resnet50-v2.h5")


Run TFLite Converter from Arachne
---------------------------------

Now, let's convert the model into a TFLite model by Arachne.
To use the TFLite Converter, we have to specify `+tools=tflite_converter` to `arachne.driver.cli`.
Available options can be seen by adding `--help`.

.. code:: bash

    python -m arachne.driver.cli +tools=tflite_converter --help


Convert with FP32 Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we will start with the simplest case.
You can convert a TF model into a TFLite mode without the post-training quantization (PTQ) by the following command.

.. code:: bash

    python -m arachne.driver.cli +tools=tflite_converter model_file=/tmp/resnet50-v2.h5 output_path=/tmp/output_fp32.tar


To check the converted model, please unpack the output TAR file and inspect the tflite model file by a model viewer like the Netron.


.. code:: bash

    tar xf /tmp/output_fp32.tar -C /tmp
    ls /tmp/model_0.tflite


Convert with Dynamic-Range or FP16 Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To convert with the dynamic range or FP16 precision, just set `dynamic_range` or `fp16` to the `tools.tflite_converter.ptq.method` option.

.. code:: bash

    python -m arachne.driver.cli +tools=tflite_converter model_file=/tmp/resnet50-v2.h5 output_path=/tmp/output_dr.tar \
        tools.tflite_converter.ptq.method=dynamic_range

    python -m arachne.driver.cli +tools=tflite_converter model_file=/tmp/resnet50-v2.h5 output_path=/tmp/output_fp16.tar \
        tools.tflite_converter.ptq.method=fp16



Convert with INT8 Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To convert with INT8 precision, we need calibrate or estimate the range of all floating-point tensors in the model.
We provide an interface to feed the dataset to be used in the calibration.
First, we have to prepare a NPY file that contains a list of `np.ndarray` which is a dataset used for calibration.


.. code:: python

    import numpy as np
    calib_dataset = []

    for image, label in val_dataset.unbatch().batch(1).take(100):
        calib_dataset.append(image.numpy())
    np.save("/tmp/calib_dataset.npy", calib_dataset)


Next, specify `int8` to the `tools.tflite_converter.ptq.method` option and pass the NPY file to the `tools.tflite_converter.ptq.representative_dataset`.

.. code:: bash

    python -m arachne.driver.cli +tools=tflite_converter model_file=/tmp/resnet50-v2.h5 output_path=/tmp/output_int8.tar \
        tools.tflite_converter.ptq.method=int8 tools.tflite_converter.ptq.representative_dataset=/tmp/calib_dataset.npy


Run TFLite Converter from Arachne Python Interface
--------------------------------------------------

The following code shows an example of using the TFLite Converter from Arachne Python interface.
The details of the API are described in :ref:`arachne.tools.tflite_converter <api-tools-tflite-converter>`.

.. code:: python

    from arachne.utils.model_utils import init_from_file, save_model
    from arachne.tools.tflite_converter import TFLiteConverter, TFLiteConverterConfig

    model_file_path = "/tmp/resnet50-v2.h5"
    input = init_from_file(model_file_path)

    cfg = TFLiteConverterConfig()

    # plz modify the config object to control the converter behavior
    # cfg.ptq.method = "FP16"

    output = TFLiteConverter.run(input, cfg)

    save_model(model=output, output_path="/tmp/output.tar")


Jupyter Notebook Link
---------------------
You can see a notebook for this tutorial `here <https://github.com/fixstars/arachne/blob/main/examples/tools/run_tflite_converter.ipynb>`_.