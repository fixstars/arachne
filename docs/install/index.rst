
Installing Arachne with Development Environment
===============================================
This page gives instructions on how to setup the arachne to be available.
It consists of two steps:

1. First download the source code from Git.
2. Build and Run Docker Container.
3. Setup Python Virutalenv by Poetry.


Expected Environment
--------------------
* OS: Ubuntu 18.04
* CUDA: 10.2
* cuDNN: 8
* TensorRT: 7.2.3
* TensorFlow: 2.4.1
* PyTorch: 1.8.0
* ONNX and ONNX Runtime: 1.6


Get Source from Git
-------------------

.. code:: bash

    git clone --recursive -b feature/v0.2 ssh://git@gitlab.fixstars.com:8022/arachne/arachne.git arachne


Setup for Development Container
-------------------------------

Method 1: Visual Studio Code (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use the Visual Studio Code (VSCode), you can use a development container via VSCode.
Please refer `the official document <https://code.visualstudio.com/docs/remote/containers>`_ to setup your VSCode.


Method 2: Docker Command (TBD)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

    cd /path/to/arachne
    docker build -t <tag name> -f docker/devel-gpu.Dockerfile docker
    docker run --rm -it -u `id -u`:`id -g` -v $PWD:/workspaces/arachne -w="/workspaces/arachne" <tag name> bash


.. attention:: TODO remap user id



Create Python Virtual Environment
---------------------------------

.. code:: bash

    # create a virtual env to /workspaces/arachne/.venv
    poetry install
    poetry shell

    # install other required packages
    ./scripts/build_tvm.sh
    ./scripts/install_torch2trt.sh


Check Arachne Works Correctly
-----------------------------

.. code:: bash

    pytest tests --forked

.. note:: `--forked` is required to make sure that the GPU memory is released for each test.
