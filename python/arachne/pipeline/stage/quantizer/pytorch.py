import itertools
from pathlib import Path
from typing import Callable, Optional

from arachne.dataset import Dataset
from arachne.pipeline.package import (
    Package,
    PackageInfo,
    PyTorchPackage,
    PyTorchPackageInfo,
    TorchScriptPackage,
    TorchScriptPackageInfo,
)
from arachne.pipeline.stage.utils import (
    get_make_dataset_from_params,
    get_preprocess_from_params,
    get_qtype_from_params,
)
from arachne.types import QType, TensorInfoDict

from .._registry import register_stage, register_stage_candidate
from ..stage import Parameter, Stage


class PyTorchQuantizer(Stage):
    @staticmethod
    def _calibration(
        qmodel,
        dataset: Dataset,
        preprocess: Callable,
        input_info: TensorInfoDict,
        calibrate_num: int,
    ):
        import torch
        import torch.nn
        import torchvision.transforms.functional

        qmodel.eval()
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for image, target in itertools.islice(dataset, calibrate_num):
                if not isinstance(image, torch.Tensor):
                    image = torchvision.transforms.functional.to_tensor(image)
                _, preprocessed = preprocess(image, input_info).get_by_index(0)
                preprocessed = torch.from_numpy(preprocessed)
                target = torch.tensor([target])
                output = qmodel(preprocessed)
                loss = criterion(output, target)

    @classmethod
    def get_name(cls) -> str:
        return "pytorch_quantizer"

    @classmethod
    def get_output_info(cls, input: PackageInfo, params: Parameter) -> Optional[PackageInfo]:
        params = PyTorchQuantizer.extract_parameters(params)
        quantize_type = params["qtype"]
        if not isinstance(input, PyTorchPackageInfo):
            return None
        if quantize_type not in (QType.FP32, QType.INT8):
            return None
        if quantize_type == QType.INT8 and not input.quantizable:
            return None
        if params["make_dataset"] is None or params["preprocess"] is None:
            return None

        return TorchScriptPackageInfo(qtype=quantize_type)

    @classmethod
    def extract_parameters(cls, params: Parameter) -> Parameter:
        quantize_type = get_qtype_from_params(params)
        samples = int(params.get("qsample", "256"))
        qbackend = params.get("qbackend")
        if qbackend is None and "_compiler_target" in params:
            target = params["_compiler_target"]
            if "x86" in target:
                qbackend = "fbgemm"
            elif "arm" in target:
                qbackend = "qnnpack"
        if qbackend is None:
            qbackend = "qnnpack"
        make_dataset, make_dataset_str = get_make_dataset_from_params(params)
        preprocess, preprocess_str = get_preprocess_from_params(params)

        return {
            "qsample": samples,
            "qtype": quantize_type,
            "qbackend": qbackend,
            "make_dataset": make_dataset,
            "make_dataset_str": make_dataset_str,
            "preprocess": preprocess,
            "preprocess_str": preprocess_str,
        }

    @classmethod
    def process(cls, input: Package, params: Parameter, output_dir: Path) -> Package:
        import torch
        import torch.nn
        import torch.quantization

        params = PyTorchQuantizer.extract_parameters(params)
        quantize_type = params["qtype"]
        samples = params["qsample"]
        qbackend = params["qbackend"]

        assert isinstance(input, PyTorchPackage)

        model = torch.load(input.dir / input.model_file)

        if quantize_type == QType.FP32:
            pass
        elif quantize_type == QType.INT8:
            make_dataset = params["make_dataset"]
            assert make_dataset is not None
            dataset = make_dataset()
            assert isinstance(dataset, Dataset)
            preprocess = params["preprocess"]
            assert preprocess is not None

            model.eval()
            model.fuse_model()
            qconfig = torch.quantization.get_default_qconfig(qbackend)
            model.qconfig = qconfig
            torch.quantization.prepare(model, inplace=True)
            PyTorchQuantizer._calibration(
                model, dataset, preprocess, input.input_info, calibrate_num=samples
            )
            torch.quantization.convert(model, inplace=True)
        else:
            raise NotImplementedError(f"{quantize_type.value} quantization is not implemented.")

        filename = "quantized.pth"
        dst_path = output_dir / filename
        input_data = [torch.randn(info.shape) for name, info in input.input_info.items()]
        scripted_model = torch.jit.trace(model, input_data).eval()
        scripted_model.save(dst_path)

        return TorchScriptPackage(
            dir=output_dir,
            input_info=input.input_info,
            output_info=input.output_info,
            qtype=quantize_type,
            model_file=Path(filename),
        )


register_stage(PyTorchQuantizer)

register_stage_candidate(PyTorchQuantizer, {"qtype": "fp32"})
register_stage_candidate(PyTorchQuantizer, {"qtype": "int8"})
