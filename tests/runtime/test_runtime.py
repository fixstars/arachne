import tempfile

import torch
import torch.onnx
import torchvision
from omegaconf.dictconfig import DictConfig

import arachne.runtime
import arachne.tools.tvm
from arachne.data import Model
from arachne.tools.tvm import TVMConfig
from arachne.utils import get_model_spec, save_model


def test_tvm_runtime():
    def compile():
        cfg = TVMConfig()
        cfg.cpu_target = "x86-64"
        cfg.composite_target = ["cpu"]

        resnet18 = torchvision.models.resnet18(pretrained=True)
        dummy_input = torch.randn(1, 3, 224, 224)
        onnx_model_path = "./resnet18.onnx"
        torch.onnx.export(resnet18, dummy_input, onnx_model_path)
        input = Model("./resnet18.onnx", spec=get_model_spec("resnet18.onnx"))
        model = arachne.tools.tvm.run(input=input, cfg=cfg)
        return model, cfg

    with tempfile.TemporaryDirectory() as tmp_dir:
        package_path = tmp_dir + "/package.tar"
        model, tvmcfg = compile()
        cfg = DictConfig({"tools": {"tvm": tvmcfg}})
        save_model(model, package_path, cfg)
        rtmodule = arachne.runtime.init(package=package_path)
        assert rtmodule
        rtmodule.benchmark()
