import itertools
from enum import Enum
from pathlib import Path
from typing import Optional

from arachne.dataset import Dataset
from arachne.logger import Logger
from arachne.pipeline.package import (
    KerasPackage,
    KerasPackageInfo,
    TF1Package,
    TF1PackageInfo,
    TF2Package,
    TF2PackageInfo,
)
from arachne.pipeline.stage.utils import (
    get_make_dataset_from_params,
    get_preprocess_from_params,
    get_qtype_from_params,
    parse_bool,
)
from arachne.runtime.package import (
    Package,
    PackageInfo,
    TFLitePackage,
    TFLitePackageInfo,
)
from arachne.types import IndexedOrderedDict, QType

from .._registry import register_stage, register_stage_candidate
from ..stage import Parameter, Stage

logger = Logger.logger()


class SetShapeMode(Enum):
    OFF = "off"
    ON = "on"
    AUTO = "auto"


class TFLiteConverter(Stage):
    @classmethod
    def get_name(cls) -> str:
        return "tflite_converter"

    @classmethod
    def get_output_info(cls, input: PackageInfo, params: Parameter) -> Optional[PackageInfo]:
        params = TFLiteConverter.extract_parameters(params)
        quantize_type = params["qtype"]
        if not isinstance(input, (KerasPackageInfo, TF1PackageInfo, TF2PackageInfo)):
            return None
        if params["qtype"] == QType.INT8_FULL:
            return None
        if params["qtype"] != QType.FP32:
            if params["make_dataset"] is None or params["preprocess"] is None:
                return None

        return TFLitePackageInfo(qtype=quantize_type, for_edgetpu=False)

    @classmethod
    def extract_parameters(cls, params: Parameter) -> Parameter:
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

    @classmethod
    def process(cls, input: Package, params: Parameter, output_dir: Path) -> Package:
        params = TFLiteConverter.extract_parameters(params)
        quantize_type = params["qtype"]
        samples = params["qsample"]
        set_shape = params["set_shape"]

        new_names = list(input.input_info.keys())
        import tensorflow as tf

        if isinstance(input, KerasPackage):
            h5_model_path = str(input.dir / input.model_file)
            model = tf.keras.models.load_model(h5_model_path)
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
        elif isinstance(input, TF1Package):
            import tensorflow.compat.v1 as tf1

            input_tensors = {name: info.shape for (name, info) in input.input_info.items()}
            converter = tf1.lite.TFLiteConverter.from_frozen_graph(
                str(input.dir / input.model_file),
                list(input.input_info.keys()),
                list(input.output_info.keys()),
                input_tensors,
            )
        elif isinstance(input, TF2Package):
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
                        logger.info(f"Set input shape: {tensor.name} <= {info.shape}")
                        new_names.append(tensor.name.split(":")[0])
                        tensor.set_shape(info.shape)

                    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

            if converter is None:
                converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        else:
            raise RuntimeError(
                "The type of input package must be TF1Package or TF2Package, "
                f"but it is {input.__class__.__name__}"
            )

        converter.allow_custom_ops = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

        if quantize_type is QType.FP32:
            pass
        elif quantize_type is QType.FP16:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quantize_type is QType.INT8:
            make_dataset = params["make_dataset"]
            assert make_dataset is not None
            dataset = make_dataset()
            assert isinstance(dataset, Dataset)
            preprocess = params["preprocess"]
            assert preprocess is not None

            def representative_dataset_gen():
                if isinstance(dataset, tf.data.Dataset):
                    for dat in dataset.take(samples):
                        preprocessed = preprocess(dat["image"])
                        yield [preprocessed]
                else:
                    import torch
                    import torchvision.transforms.functional

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

        return TFLitePackage(
            dir=output_dir,
            input_info=new_input_info,
            output_info=input.output_info,
            qtype=quantize_type,
            for_edgetpu=False,
            model_file=Path(filename),
        )


register_stage(TFLiteConverter)

register_stage_candidate(TFLiteConverter, {"qtype": "fp32"})
register_stage_candidate(TFLiteConverter, {"qtype": "int8"})
