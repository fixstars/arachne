import tvm
from tvm.driver import tvmc
import tensorflow as tf
import tempfile
import os

def compile(model, target_device, pipeline, output_dir):

    if isinstance(model, tf.keras.Model):
        model_format = 'keras'
        # NOTE assume 1 input layer
        input_layer = model.get_layer(index=0)
        config = input_layer.get_config()
        input_shape = tuple([1] + list(config['batch_input_shape'][1:]))
        input_shape_dict = {
            config['name'] : input_shape
        }

        if pipeline == 'tvm':
            with tempfile.TemporaryDirectory() as tmp_dir:
                h5_path = os.path.join(tmp_dir, model.name + ".h5")
                model.save(h5_path)

                tvm_model = tvmc.frontends.load_model(h5_path, model_format, input_shape_dict)

                if target_device == 'jetson-nano':
                    target_host = "llvm -keys=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon"
                    target = "tensorrt --remove_no_mac_subgraphs, " + target_host
                    cross_arm = "aarch64-linux-gnu-gcc"

                package_path = os.path.join(output_dir, model.name + ".tar")
                tvmc.compiler.compile_model(
                        tvm_model,
                        target,
                        package_path=package_path,
                        cross=cross_arm,
                        dump_code='relay',
                        target_host=target_host,
                        desired_layout=None)

                return (package_path, package_path + ".relay")
