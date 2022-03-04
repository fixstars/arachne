Tensorflow Object Detection API
===============================


Install TF Object Dtection API
------------------------------

.. code:: bash

    git clone https://github.com/tensorflow/models.git


.. code:: bash

    sudo apt-get install -y protobuf-compiler
    protc --version

.. code:: bash

    cd models/research
    protoc object_detection/protos/*.proto --python_out=.


.. code:: bash

    cp object_detection/packages/tf2/setup.py .
    python -m pip install --use-feature=2020-resolver .


To test the API works correctly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: bash

    python object_detection/builders/model_builder_tf2_test.py



To downgrade TF version
^^^^^^^^^^^^^^^^^^^^^^^

Add package dependencies for specifying the tensorflow and tf-models-official version.

.. code:: python

    # models/research/setup.py
    REQUIRED_PACKAGES = [
        ...
        'tensorflow==2.5.2',
        'tf-models-official~=2.5',
        ...
    ]



An example of TF2 object detection API with arachne
---------------------------------------------------

TODO: put a link to a jupyter notebook



How to evaluate the models converted or compiled by arachne
-----------------------------------------------------------

Step 1: convert into the tflite-friendly model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    python -m object_detection.export_tflite_graph_tf2 \
    --pipeline_config_path /path/to/model_dir/pipeline.config \
    --trained_checkpoint_dir /path/to/model_dir/checkpoint \
    --max_detections <max detection num: e.g., 100> \
    --output_directory /path/to/output_dir


Step 2: apply a pipeline/tool to the tflite-friendly model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    # An example: compile a model after converting it into a tflite model.

    python -m arachne.driver.pipeline \
        input=/path/to/output_dir/saved_model \
        output=/path/to/arachne_output.tar \
        pipeline=[tflite_converter,tvm] \
        +tvm_target=dgx-1


Step 3: evaluate the output model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please confirm your current working directory is `/path/to/models/research/object_detection/arachne`.
You have to setup the `eval_input_reader` configuration in advance.
Typically, you need to fill the correct path to `label_map_path` and `input_path`.
For example, an example of the `eval_input_reader` setting looks like below.

.. code::

    eval_input_reader {
        label_map_path: "/path/to/models/research/object_detection/data/mscoco_label_map.pbtxt"
        shuffle: false
        num_epochs: 1
        tf_record_input_reader {
            input_path: "/path/to/datasets/coco/tfrecord/coco_val.record-?????-of-00050"
        }
    }

Then, you can evaluate the model by the following command.

.. code:: bash


    python eval.py \
        arachne_output_path=/path/to/arachne_output.tar \
        pipeline_config_path=/path/to/model_dir/pipeline.config \
        checkpoint_dir_path=/path/to/model_dir/checkpoint

If you evaluate the model on a remote device, please specify `rpc_host=<hostname or IP address>` and `rpc_port=<port number>`.