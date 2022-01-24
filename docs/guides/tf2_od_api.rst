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

