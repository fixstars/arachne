YOLOX
=====

In this user guide, compile YOLOX-s and run compiled model with arachne.

.. attention:: TODO: change repository url after public release on github.

Setup
#####

Build arachne container image.

.. code:: bash

    git clone ssh://git@gitlab.fixstars.com:8022/arachne/arachne.git
    cd arachne
    docker build -f docker/devel-gpu.Dockerfile -t arachne  --build-arg GITLAB_ACCESS_TOKEN=XXXXXXXXXXXX .
    cd ..

Clone YOLOX repository and open the directory in vscode and run :code:`Remote-Containers: Reopen in Container` from command pallet.

.. code:: bash

    git clone ssh://git@gitlab.fixstars.com:8022/arachne/models/YOLOX.git
    code ./YOLOX

| The following steps are done inside the docker container.
| Run :code:`setup_arachne.sh` to setup virtual environment and install requirements for YOLOX.

.. code:: bash

    cd /workspaces/YOLOX
    ./setup_arachne.sh

Activate virtual environment.

.. code:: bash

    source ~/arachne_src/.venv/bin/activate


Compile models using arachne
#########################

Export :code:`yolox_s.pth` in onnx format.

.. code:: bash

    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
    python tools/export_onnx.py --output-name yolox_s.onnx -n yolox-s -c yolox_s.pth

Compile yolox-s with :code:`arachne.tools`.
You can select :code:`cpu, cuda, or tensorrt` for :code:`tools.tvm.composite_target`.
Model spec is defined in :code:`yaml/yolox-s.yml`:

.. code:: yaml

    inputs:
    - dtype: float32
        name: images
        shape:
        - 1
        - 3
        - 640
        - 640
    outputs:
    - dtype: float32
        name: output
        shape:
        - 1
        - 8400
        - 85

.. code:: bash

    python -m arachne.tools.tvm \
        input=./yolox_s.onnx \
        input_spec=./yaml/yolox_s.yml \
        output=./yolox_s.tar \
        tools.tvm.composite_target=[tensorrt,cpu]

Run compiled model
##################

Run compiled model using :code:`arachne.runtime.module`.

.. code:: python

    import cv2
    import torch
    import numpy as np

    from yolox.utils import postprocess as util_postprocess
    from yolox.utils import demo_postprocess, vis
    from yolox.data.datasets import COCO_CLASSES
    import arachne.runtime

    def preprocess(img):
        resized_img = cv2.resize(orig_img, (640, 640))
        resized_img = resized_img.transpose(2, 0, 1)
        resized_img = resized_img[np.newaxis, :, :, :]
        return resized_img


    def postprocess(outputs):
        outputs = demo_postprocess(outputs, (640, 640))
        outputs = util_postprocess(outputs, 80, conf_thre=0.40, nms_thre=0.45)
        output = outputs[0]
        bboxes = output[:, 0:4]
        ratio = (640 / orig_img.shape[0], 640 / orig_img.shape[1])
        bboxes[:, 0] /= ratio[1]
        bboxes[:, 1] /= ratio[0]
        bboxes[:, 2] /= ratio[1]
        bboxes[:, 3] /= ratio[0]
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        return bboxes, scores, cls


    orig_img = cv2.imread("./assets/dog.jpg")
    input_data = preprocess(orig_img)
    rtmod = arachne.runtime.init(package_tar="./yolox_s.tar")
    rtmod.set_input(0, input_data)
    rtmod.run()
    outputs = rtmod.get_output(0)
    outputs = torch.from_numpy(outputs)
    bboxes, scores, cls = postprocess(outputs)
    vis_res = vis(orig_img, bboxes, scores, cls, conf=0.40, class_names=COCO_CLASSES)
    cv2.imwrite("result.jpg", vis_res)

Evaluate compiled model
#######################

Run arachne RPC server in other shell.

.. code:: bash

    python -m arachne.runtime.rpc.server --port 5051 --runtime tvm

Run evaluate script.

.. code:: bash

    python tools/eval.py \
        -n yolox-s \
        -c yolox_s.pth \
        -b 1 \
        -d 1 \
        --conf 0.001 \
        --arachne-package-path "yolox_s.tar" \
        --arachne-rpc-host localhost \
        --arachne-rpc-port 5051 \
        data_dir /datasets/COCO data_num_workers 0

.. note:: You need to set data_num_workers to 0.
    This is because if you run the data loader in a multi-process, multiple clients are created, and may cause RPC communication failure.

Evaluation results are the following:

.. code::

    Average forward time: 40.06 ms, Average NMS time: 8.64 ms, Average inference time: 48.70 ms
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.405
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.593
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.438
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.233
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.448
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.541
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.326
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.531
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.574
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.366
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.635
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.724

| :code:`ArachneCOCOEvaluator` is an implementation of inference execution by arachne.
| See also `arachne_example.ipyenb <https://gitlab.fixstars.com/arachne/models/YOLOX/-/blob/main/arachne_example.ipyenb>`_ in YOLOX repository.