import tempfile
from pathlib import Path

from arachne.device import get_target
from arachne.pipeline.package.frontend import make_tf1_package_from_concrete_func
from arachne.pipeline.runner import make_pipeline_candidate
from arachne.pipeline.stage.converter import TFLiteConverter


def test_make_pipeline_candidate():
    with tempfile.TemporaryDirectory() as tmp_dir:
        import tensorflow as tf

        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.float32),
            ]
        )
        def add(x, y):
            return x + y

        concrete_func = add.get_concrete_function()

        pkg = make_tf1_package_from_concrete_func(concrete_func, Path(tmp_dir))
        target = get_target("host")
        assert target is not None
        pipelines = make_pipeline_candidate(pkg, [target])

        # tvm_compiler, tflite_converter:fp32 -> tvm_compiler, auto_scheduler -> tvm_compiler,
        # tflite_converter:fp32 -> auto_scheduler -> tvm_compiler
        assert len(pipelines) == 4

        exclude = set()
        exclude.add(TFLiteConverter)
        pipelines = make_pipeline_candidate(pkg, [target], {}, exclude)

        # tvm_compiler, auto_scheduler -> tvm_compiler
        assert len(pipelines) == 2
