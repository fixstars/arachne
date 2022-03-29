TF-TRT
======

TensorFlow with TensorRT (TF-TRT) is a Tensorflow integration for optimizing Tensorflow models to execute them with TensorRT.


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
    model.fit(train_dataset, validation_data=val_dataset, epochs=5)

    model.evaluate(val_dataset)

    model.save("/tmp/saved_model")


Run TF-TRT from Arachne
-----------------------

Now, let's optimize the model with TF-TRT by Arachne.
To use the TF-TRT, we have to specify `+tools=tftrt` to `arachne.driver.cli`.
Available options can be seen by adding `--help`.

.. code:: bash

    python -m arachne.driver.cli +tools=tftrt --help


Optimize with FP32 Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, we will start with the simplest case.
You can optimize a TF model with FP32 precision by the following command.

.. code:: bash

    python -m arachne.driver.cli +tools=tftrt input=/tmp/saved_model output=/tmp/output.tar


Optimize with FP16 precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To optimize with FP16 precision, specify `FP16` to the `tools.tftrt.precision_mode` option.

.. code:: bash

    python -m arachne.driver.cli +tools=tftrt input=/tmp/saved_model output=/tmp/output.tar tools.tftrt.precision_mode=FP16


Optimize with INT8 Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To convert with INT8 precision, we need calibrate or estimate the range of all floating-point tensors in the model.
We provide an interface to feed the dataset to be used in the calibration.
First, we have to prepare a NPY file that contains a list of `np.ndarray` which is a dataset used for calibration.

.. code:: python

    import numpy as np
    calib_dataset = []

    for image, label in val_dataset.unbatch().batch(1).take(100):
        calib_dataset.append(image.numpy())
    np.save("/tmp/calib_dataset.npy", calib_dataset)

Next, specify `INT8` to the `tools.tftrt.precision_mode` option and pass the NPY file to the `tools.tftrt.representative_dataset`.

.. code:: bash

    python -m arachne.driver.cli +tools=tftrt input=/tmp/saved_model output=/tmp/output.tar \
        tools.tftrt.precision_mode=INT8 \
        tools.tftrt.representative_dataset=/tmp/calib_dataset.npy


Run TF-TRT from Arachne Python Interface
----------------------------------------

The following code shows an example of using the tool from Arachne Python interface.
The details are described in :ref:`arachne.tools.tftrt <api-tools-tftrt>`.

.. code:: python

    from arachne.data import Model
    from arachne.utils.model_utils import get_model_spec, save_model
    from arachne.tools.tftrt import TFTRT, TFTRTConfig

    model_file_path = "/tmp/saved_model"
    input = Model(path=model_file_path, spec=get_model_spec(model_file_path))

    cfg = TFTRTConfig()

    # cfg.precision_mode = "FP16"

    output = TFTRT.run(input, cfg)

    save_model(model=output, output_path="/tmp/output.tar")

Jupyter Notebook Link
---------------------
You can see a notebook for this tutorial `here <https://github.com/fixstars/arachne/blob/main/examples/tools/run_tftrt.ipynb>`_.