import tarfile
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse

from tvm.contrib.download import download as tvm_download

from arachne.pipeline.package import (
    DarknetPackage,
    KerasPackage,
    PyTorchPackage,
    Tf1Package,
    Tf2Package,
)
from arachne.pipeline.package.torchscript import TorchScriptPackage
from arachne.types.indexed_ordered_dict import TensorInfoDict
from arachne.types.qtype import QType
from arachne.types.tensor_info import TensorInfo


def download(model_urls: Union[List[str], str], output_dir: Path) -> List[Path]:
    if isinstance(model_urls, str):
        model_urls = [model_urls]

    outputs: List[Path] = []
    for model_url in model_urls:
        output_path = output_dir / Path(urlparse(model_url).path).name
        tvm_download(model_url, str(output_path))
        outputs.append(output_path)

    return outputs


def make_tf1_package(
    model_url: str, input_info: TensorInfoDict, output_info: TensorInfoDict, output_dir: Path
) -> Tf1Package:
    outputs = download(model_url, output_dir)

    return Tf1Package(
        dir=output_dir,
        input_info=input_info,
        output_info=output_info,
        model_file=Path(outputs[0].name),
    )


def make_tf1_package_from_concrete_func(cfunc, output_dir: Path) -> Tf1Package:
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    frozen_model, graph_def = convert_variables_to_constants_v2_as_graph(cfunc)

    tf.io.write_graph(
        graph_or_graph_def=graph_def,
        logdir=output_dir,
        name="frozen_graph.pb",
        as_text=False,
    )

    input_info = TensorInfoDict()
    for input in frozen_model.inputs:
        name = input.name.replace(":0", "")
        input_info[name] = TensorInfo(shape=input.shape.as_list(), dtype=input.dtype.name)

    output_info = TensorInfoDict()
    for output in frozen_model.outputs:
        name = output.name.replace(":0", "")
        output_info[name] = TensorInfo(shape=output.shape.as_list(), dtype=output.dtype.name)

    return Tf1Package(
        dir=output_dir,
        input_info=input_info,
        output_info=output_info,
        model_file=Path("frozen_graph.pb"),
    )


def make_tf2_package(
    model_url: str,
    input_info: TensorInfoDict,
    output_info: TensorInfoDict,
    output_dir: Path,
    model_dir: str = "saved_model",
) -> Tf2Package:
    outputs = download(model_url, output_dir)

    with tarfile.open(outputs[0], "r:gz") as tar:
        tar.extractall(output_dir)

    return Tf2Package(
        dir=output_dir,
        input_info=input_info,
        output_info=output_info,
        model_dir=Path(model_dir),
    )


# NOTE: model should be a tf.Module
def make_tf2_package_from_module(
    model, input_info: TensorInfoDict, output_info: TensorInfoDict, output_dir: Path
) -> Tf2Package:
    import tensorflow as tf

    model_path = output_dir / "saved_model"
    tf.saved_model.save(model, str(model_path))

    return Tf2Package(
        dir=output_dir,
        input_info=input_info,
        output_info=output_info,
        model_dir=Path(model_path.name),
    )


# NOTE: model should be a tf.Module
def make_keras_package_from_module(model, output_dir: Path) -> KerasPackage:

    h5_path = output_dir / (model.name + ".h5")
    model.save(h5_path)

    input_info = TensorInfoDict()
    for inp in model.inputs:
        input_info[inp._name] = TensorInfo([1] + inp.shape.as_list()[1:])

    output_info = TensorInfoDict()
    for out in model.outputs:
        output_info[out._name] = TensorInfo([1] + out.shape.as_list()[1:])

    return KerasPackage(
        dir=output_dir,
        input_info=input_info,
        output_info=output_info,
        model_file=Path(h5_path),
    )


# NOTE: model_def should be a torch.nn.Module
def make_pytorch_package(
    model_url: str,
    input_info: TensorInfoDict,
    output_info: TensorInfoDict,
    output_dir: Path,
    model_def=None,
) -> PyTorchPackage:
    import torch

    outputs = download(model_url, output_dir)

    if model_def:
        model = model_def
        model.load_state_dict(torch.load(outputs[0]))
        model_path = output_dir / "model.pickle"
        torch.save(model, model_path)
    else:
        model_path = outputs[0]

    return PyTorchPackage(
        dir=output_dir,
        input_info=input_info,
        output_info=output_info,
        quantizable=False,
        model_file=Path(model_path.name),
    )


# NOTE: model should be a torch.nn.Module
def make_pytorch_package_from_module(
    model,
    input_info: TensorInfoDict,
    output_info: TensorInfoDict,
    output_dir: Path,
) -> PyTorchPackage:
    import torch

    model_path = output_dir / "model.pickle"
    torch.save(model, model_path)

    return PyTorchPackage(
        dir=output_dir,
        input_info=input_info,
        output_info=output_info,
        quantizable=False,
        model_file=Path(model_path.name),
    )


def make_torchscript_package_from_script_module(
    script,
    input_info: TensorInfoDict,
    output_info: TensorInfoDict,
    output_dir: Path,
    qtype: QType = QType.FP32,
) -> TorchScriptPackage:
    model_path = output_dir / "model.pth"
    script.save(model_path)

    return TorchScriptPackage(
        dir=output_dir,
        input_info=input_info,
        output_info=output_info,
        model_file=Path(model_path.name),
        qtype=qtype,
    )


def make_darknet_package(
    cfg_url: str,
    weight_url: str,
    input_info: TensorInfoDict,
    output_info: TensorInfoDict,
    output_dir: Path,
) -> DarknetPackage:
    outputs = download([cfg_url, weight_url], output_dir)

    return DarknetPackage(
        dir=output_dir,
        input_info=input_info,
        output_info=output_info,
        cfg_file=Path(outputs[0].name),
        weight_file=Path(outputs[1].name),
    )
