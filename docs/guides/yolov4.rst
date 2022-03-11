YOLOv4
======

In this user guide, compile YOLOv4 and run compiled model with arachne.

.. attention:: TODO: change repository url after public release on github.

Setup
#####

Build arachne container image.

.. code:: bash

    git clone git@github.com:fixstars/arachne.git arachne
    git clone --recursive git@github.com:fixstars/tvm.git arachne/3rdparty/tvm
    cd arachne
    docker build -f docker/devel-gpu.Dockerfile -t arachne  --build-arg GITLAB_ACCESS_TOKEN=XXXXXXXXXXXX .
    cd ..

Clone YOLOv4 repository and open the directory in vscode and run :code:`Remote-Containers: Reopen in Container` from command pallet.

.. code:: bash

    git clone -b compile_with_arachne ssh://git@gitlab.fixstars.com:8022/arachne/models/pytorch-YOLOv4.git
    code ./pytorch-YOLOv4

| The following steps are done inside the docker container.
| Run :code:`setup_arachne.sh` to setup virtual environment and install requirements for YOLOv4.

.. code:: bash

    cd /workspaces/pytorch-YOLOv4
    ./setup_arachne.sh

Activate virtual environment.

.. code:: bash

    source ~/arachne_src/.venv/bin/activate


Compile models using arachne
############################

Export :code:`yolov4.weights` in onnx format.

.. code:: bash

    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
    python demo_darknet2onnx.py ./cfg/yolov4.cfg ./data/coco.names ./yolov4.weights ./data/dog.jpg 1 yolov4.onnx

Compile yolov4 with :code:`arachne.tools.tvm`.
You can set :code:`cpu, cuda` for :code:`tools.tvm.composite_target`.
Currently you cannot specify tensorrt due to a `problem with the tvm compiler <https://gitlab.fixstars.com/arachne/arachne/-/issues/150>`_.

Model spec is defined in :code:`yaml/yolov4.yml`:

.. code:: yaml

    inputs:
    - dtype: float32
        name: input
        shape:
        - 1
        - 3
        - 608
        - 608
    outputs:
    - dtype: float32
        name: boxes
        shape:
        - 1
        - 22743
        - 1
        - 4
    - dtype: float32
        name: confs
        shape:
        - 1
        - 22743
        - 80

.. code:: bash

    python -m arachne.driver.cli \
    +tools=tvm \
    input=./yolov4.onnx \
    input_spec=./yaml/yolov4.yml \
    output=./yolov4.tar \
    tools.tvm.composite_target=[cuda]

Run compiled model
##################

Run compiled model using :code:`arachne.runtime.module`.

.. code:: python

    import sys
    import onnx
    import os
    import argparse
    import numpy as np
    import cv2
    import onnxruntime
    import arachne
    from tool.utils import *
    from tool.darknet2onnx import *
    import arachne.runtime


    def detect(package_tar, image_path, namesfile, input_size):
        image_src = cv2.imread(image_path)
        rtmod = arachne.runtime.init(package_tar=package_tar)
        IN_IMAGE_H, IN_IMAGE_W = input_size

        # Input
        resized = cv2.resize(
            image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR
        )
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        print("Shape of the network input: ", img_in.shape)

        # Compute
        rtmod.set_input(0, img_in)
        rtmod.run()
        outputs = [rtmod.get_output(i) for i in range(2)]
        boxes = post_processing(img_in, 0.4, 0.6, outputs)

        class_names = load_class_names(namesfile)
        return plot_boxes_cv2(image_src, boxes[0], savename="result.jpg", class_names=class_names)

    result = detect(
        "./yolov4.tar",
        "./data/dog.jpg",
        "./data/coco.names",
        (608, 608),
    )
    cv2.imwrite("result.jpg", result)


Evaluate compiled model
#######################

Run arachne RPC server in other shell.

.. code:: bash

    python -m arachne.runtime.rpc.server --port 5051 --runtime tvm

Run evaluate script.

.. code:: bash

    python evaluate_on_coco.py \
        -g 1 \
        -dir /datasets/COCO/val2017 \
        -gta /datasets/COCO/annotations/instances_val2017.json \
        -w yolov4.weights \
        -c cfg/yolov4.cfg  \
        --arachne-package-path yolov4.tar \
        --arachne-rpc-host localhost \
        --arachne-rpc-port 5051 \

Evaluation results are the following:

.. code::

    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.449
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.668
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.493
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.295
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.505
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.560
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.340
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.522
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.541
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.377
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.592
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.673

| See also `arachne_example.ipyenb <https://gitlab.fixstars.com/arachne/models/pytorch-YOLOv4/-/blob/compile_with_arachne/arachne_example.ipynb>`_ in pytorch-YOLOv4 repository.
