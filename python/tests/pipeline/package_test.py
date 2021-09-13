import tempfile
from pathlib import Path

from arachne.pipeline.package.frontend import (
    make_keras_package_from_module,
    make_tf1_package_from_concrete_func,
    make_torchscript_package_from_script_module,
)
from arachne.runtime.indexed_ordered_dict import IndexedOrderedDict, TensorInfoDict
from arachne.runtime.package import import_package
from arachne.runtime.tensor_info import TensorInfo


def test_export_import_torch_script_pkg():

    import torch
    from torchvision import models

    with tempfile.TemporaryDirectory() as tmp_dir:

        model = models.resnet18(pretrained=True)
        input_spec = [TensorInfo(shape=[1, 3, 224, 224])]

        # Make an input package
        input_info = {
            f"input{n}": TensorInfo(shape=ispec.shape, dtype=ispec.dtype)
            for n, ispec in enumerate(input_spec)
        }

        inps = tuple(torch.zeros(i.shape) for i in input_spec)
        model.eval()
        script_model = torch.jit.trace(model.forward, inps).eval()

        output_info: TensorInfoDict = IndexedOrderedDict()
        for i, t in enumerate(script_model(*inps)):
            output_info["output" + str(i)] = TensorInfo(
                shape=list(t.shape), dtype=str(t.dtype).split(".")[-1]
            )

        pkg = make_torchscript_package_from_script_module(
            script_model, input_info, output_info, Path(tmp_dir)
        )

        export_pkg_path = Path(tmp_dir + "/exported.tar")
        pkg.export(export_pkg_path)

        import_dir = Path(tmp_dir + "/imported")
        pkg2 = import_package(export_pkg_path, import_dir)

        assert pkg.input_info == pkg2.input_info
        assert pkg.output_info == pkg2.output_info
        assert pkg.model_file == pkg2.model_file


def test_export_import_tf1_pkg():

    import tensorflow as tf

    with tempfile.TemporaryDirectory() as tmp_dir:

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.float32),
            ]
        )
        def add(x, y):
            return x + y

        concrete_func = add.get_concrete_function()

        pkg = make_tf1_package_from_concrete_func(concrete_func, tmp_dir)

        export_pkg_path = Path(tmp_dir + "/exported.tar")
        pkg.export(export_pkg_path)

        import_dir = Path(tmp_dir + "/imported")
        pkg2 = import_package(export_pkg_path, import_dir)

        assert pkg.input_info == pkg2.input_info
        assert pkg.output_info == pkg2.output_info
        assert pkg.model_file == pkg2.model_file


def test_export_import_keras_pkg():
    import tensorflow as tf

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = tf.keras.applications.mobilenet.MobileNet()
        pkg = make_keras_package_from_module(model, Path(tmp_dir))

        export_pkg_path = Path(tmp_dir + "/exported.tar")
        pkg.export(export_pkg_path)

        import_dir = Path(tmp_dir + "/imported")
        pkg2 = import_package(export_pkg_path, import_dir)

        assert pkg.input_info == pkg2.input_info
        assert pkg.output_info == pkg2.output_info
        assert pkg.model_file == pkg2.model_file
