
Installing Arachne
==================

Getting the Source Code
-----------------------

.. code:: bash

    $ git clone git@github.com:fixstars/arachne.git arachne
    $ git clone --recursive git@github.com:fixstars/tvm.git arachne/3rdparty/tvm


Expected Environment
--------------------
Before you begin to use the Arachne, review the expected environment given below.
Or you can create such environment by using a docker image and a virtual environment described in the following section.

* Arch: X86-64
* OS: Ubuntu 18.04
* JetPack 4.6 Compatible Environment:

  * CUDA: 10.2
  * cuDNN: 8.2.1
  * TensorRT: 8.0.1
* DNN libraries:

  * TensorFlow: 2.6.3
  * PyTorch: 1.8.0
  * ONNX and ONNX Runtime: 1.10.0
  * and so on.


Run a Dev Container
----------------------

We provide a dockerfile for development with GPUs.

Docker Requirements
^^^^^^^^^^^^^^^^^^^

1. `Install Docker <https://docs.docker.com/get-docker/>`_ on your host machine.

2. For GPU support, `install NVIDIA Docker <https://github.com/NVIDIA/nvidia-docker>`_


Start a Docker Container from CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To start a configured dev container, use the following script:

.. code:: bash

    $ pwd
    /path/to/arachne

    $ ./scripts/docker_launch.sh

The script automatically setups the working directory and the user in the container to keep the same permission on your host machine.


Start a Docker Container from Visual Stuido Code (VSCode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For VSCode users, we prepare a dev container configuration file (i.e., `devcontainer.json`) for launching a dev container.
When you open the `arachne` project directory by VSCode, you will see a pop-up to reopen in container.

Please refer to `the official document <https://code.visualstudio.com/docs/remote/containers>`_ for enabling the Remote Development extension pack.
`The reference of devcontainer.json <https://code.visualstudio.com/docs/remote/devcontainerjson-reference>`_ is also helpful to customize your configuration.


Create a Virtual Environment
----------------------------

Arachne leverages various excelent python packages.
To install them with isolating from the system, we recommend to use a virtual environment.
In Arachne, we choose `Poetry <https://python-poetry.org/docs/>`_ as a python package manager that can install dependent packages into a virtual environment.

.. code:: bash

    $ poetry install  # install dependent packages to /path/to/arachne/.venv
    $ poetry shell  # activate the virtual environmen with spawning a new shell
    $ ./scripts/install_tvm.sh
    $ ./scripts/install_torch2trt.sh


To install the TVM and the torch2trt python package, we use specific installation scripts.
To use the TVM python package, we have to build the TVM shared library (i.e, `libtvm.so`).
The script performs a build and then install a python package.
To customize the build configuration, please refer to `config.cmake <https://github.com/apache/tvm/blob/main/cmake/config.cmake>`_ and modify the installation script.

The torch2trt package requires the `tensorrt` package, but this is installed by `apt-get install python3-libnvinfer` and is one of the system-site packages.
Poetry cannot handle the system-site packages in dependency resolution, so that we decide to install the package out of the package management system.



Test Arachne
------------

.. code:: bash

    pytest tests --forked

.. note:: `--forked` is required to make sure that the GPU memory is released for each test.
