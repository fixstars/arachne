Device: Setup device
====================

:code:`arachne` currently supports the following devices and environments.

+------------------------+-------------------------+-------------+------------+
| Device                 | Environment             | onnxruntime | Tensorflow |
+========================+=========================+=============+============+
| Jetson                 | JetPack 4.6             | 1.10.0      | 2.6.2      |
| (nano, xavier-nx, tx2) +-------------------------+-------------+------------+
|                        | JetPack 4.5.1           | 1.6.0       | 2.5.0      |
+------------------------+-------------------------+-------------+------------+
| Raspberry Pi4          | Ubuntu 20.04 LTS 64-bit | 1.10.0      | 2.6.3      |
+------------------------+-------------------------+-------------+------------+

Create virtual env and install required packages
------------------------------------------------

| Here is an example setup on jetson-xavier-nx with JetPack 4.6.
| Clone repository and run install script.
| The install script creates a python virtual environment using poetry, installs the necessary packages and builds TVM runtime.

.. code:: shell

    git clone --recursive https://gitlab.fixstars.com/arachne/arachne.git
    cd arachne/device
    ./jp46/install.sh



You can activate virtual environment after installation is complete.

.. code:: shell

    source ./jp46/.venv/bin/activate
