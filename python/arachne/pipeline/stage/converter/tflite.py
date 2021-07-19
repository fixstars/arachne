import itertools
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms.functional
from arachne.pipeline.package import (
    Package,
    PackageInfo,
    Tf1Package,
    Tf1PackageInfo,
    Tf2Package,
    Tf2PackageInfo,
    TfLitePackage,
    TfLitePackageInfo,
)
from arachne.pipeline.stage.utils import (
    get_make_dataset_from_params,
    get_preprocess_from_params,
    get_qtype_from_params,
    parse_bool,
)
from arachne.types import ArachneDataset, IndexedOrderedDict, QType

from .._registry import register_stage, register_stage_candidate
from ..stage import Parameter, Stage


class SetShapeMode(Enum):
    OFF = "off"
    ON = "on"
    AUTO = "auto"


class TfLiteConverter(Stage):
    @staticmethod
    def get_name() -> str:
        return "tflite_converter"

    @staticmethod
    def get_output_info(input: PackageInfo, params: Parameter) -> Optional[PackageInfo]:
        quantize_type = get_qtype_from_params(params)
        if not isinstance(input, (Tf1PackageInfo, Tf2PackageInfo)):
            return None
        if quantize_type == QType.INT8_FULL:
            return None

        return TfLitePackageInfo(qtype=quantize_type, for_edgetpu=False)

    @staticmethod
    def extract_parameters(params: Parameter) -> Parameter:
        quantize_type = get_qtype_from_params(params)
        samples = int(params.get("qsample", "256"))
        make_dataset, make_dataset_str = get_make_dataset_from_params(params)
        preprocess, preprocess_str = get_preprocess_from_params(params)
        set_shape_str = params.get("set_shape", "auto")
        if set_shape_str == "auto":
            set_shape = SetShapeMode.AUTO
        elif parse_bool(set_shape_str):
            set_shape = SetShapeMode.ON
        else:
            set_shape = SetShapeMode.OFF

        return {
            "qsample": samples,
            "qtype": quantize_type,
            "make_dataset": make_dataset,
            "make_dataset_str": make_dataset_str,
            "preprocess": preprocess,
            "preprocess_str": preprocess_str,
            "set_shape": set_shape,
        }

    @staticmethod
    def process(input: Package, params: Parameter, output_dir: Path) -> Package:
        params = TfLiteConverter.extract_parameters(params)
        quantize_type = params["qtype"]
        samples = params["qsample"]
        set_shape = params["set_shape"]

        new_names = list(input.input_info.keys())
        if isinstance(input, Tf1Package):
            import tensorflow.compat.v1 as tf

            input_tensors = {name: info.shape for (name, info) in input.input_info.items()}
            converter = tf.lite.TFLiteConverter.from_frozen_graph(
                str(input.dir / input.model_file),
                list(input.input_info.keys()),
                list(input.output_info.keys()),
                input_tensors,
            )
        elif isinstance(input, Tf2Package):
            import tensorflow as tf

            saved_model_path = str(input.dir / input.model_dir)
            converter = None
            if set_shape is not SetShapeMode.OFF:
                saved_model = tf.saved_model.load(saved_model_path)
                if not saved_model:
                    raise RuntimeError(f"Failed load saved_model from: {saved_model_path}")

                concrete_func = saved_model.signatures[
                    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                ]
                input_tensors = [
                    tensor
                    for tensor in concrete_func.inputs
                    if tensor.dtype != tf.resource and "unknown" not in tensor.name
                ]
                if set_shape is SetShapeMode.ON or any(
                    map(lambda tensor: None in tensor.shape, input_tensors)
                ):
                    new_names.clear()
                    for tensor, info in zip(input_tensors, input.input_info.values()):
                        new_names.append(tensor.name.split(":")[0])
                        tensor.set_shape(info.shape)

                    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

            if converter is None:
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        else:
            raise RuntimeError(
                f"The type of input package must be Tf1Package or Tf2Package, but it is {input.__class__.__name__}"
            )

        converter.allow_custom_ops = True

        if quantize_type is QType.FP32:
            pass
        elif quantize_type is QType.FP16:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
        elif quantize_type is QType.INT8:
            make_dataset = params["make_dataset"]
            assert make_dataset is not None
            dataset = make_dataset()
            assert isinstance(dataset, ArachneDataset)
            preprocess = params["preprocess"]
            assert preprocess is not None

            def representative_dataset_gen():
                for image, _ in itertools.islice(dataset, samples):
                    if not isinstance(image, torch.Tensor):
                        image = torchvision.transforms.functional.to_tensor(image)
                    preprocessed = preprocess(image, input.input_info)
                    yield [preprocessed[input.input_info.get_by_index(0)[0]]]

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset_gen
        elif quantize_type is QType.INT8_FULL:
            # converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # converter.representative_dataset = functools.partial(
            #     representative_dataset_gen, experiment=experiment, samples=samples)
            # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # converter.inference_input_type = tf.uint8
            raise NotImplementedError("Not implemented for full-integer quantization")

        tflite_quant_model = converter.convert()

        filename = "quantized.tflite"
        dst_path = output_dir / filename
        with open(dst_path, "wb") as w:
            w.write(tflite_quant_model)

        new_input_info = IndexedOrderedDict(
            [(new_name, info) for new_name, info in zip(new_names, input.input_info.values())]
        )

        return TfLitePackage(
            dir=output_dir,
            input_info=new_input_info,
            output_info=input.output_info,
            qtype=quantize_type,
            for_edgetpu=False,
            model_file=Path(filename),
        )


register_stage(TfLiteConverter)

register_stage_candidate(TfLiteConverter, {"qtype": "fp32"})
register_stage_candidate(TfLiteConverter, {"qtype": "int8"})
