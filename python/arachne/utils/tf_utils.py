import tensorflow as tf

from ..data import ModelSpec, TensorSpec


def make_tf_gpu_usage_growth():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def get_tflite_model_spec(model_path: str) -> ModelSpec:
    inputs = []
    outputs = []
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for inp in input_details:
        inputs.append(
            TensorSpec(name=inp["name"], shape=inp["shape"].tolist(), dtype=inp["dtype"].__name__)
        )
    for out in output_details:
        outputs.append(
            TensorSpec(name=out["name"], shape=out["shape"].tolist(), dtype=out["dtype"].__name__)
        )
    return ModelSpec(inputs=inputs, outputs=outputs)


def get_keras_model_spec(model_path: str) -> ModelSpec:
    inputs = []
    outputs = []
    model = tf.keras.models.load_model(model_path)
    for inp in model.inputs:  # type: ignore
        shape = [-1 if x is None else x for x in inp.shape]
        inputs.append(TensorSpec(name=inp.name, shape=shape, dtype=inp.dtype.name))
    for out in model.outputs:  # type: ignore
        shape = [-1 if x is None else x for x in out.shape]
        outputs.append(TensorSpec(name=out.name, shape=shape, dtype=out.dtype.name))
    del model
    return ModelSpec(inputs=inputs, outputs=outputs)


def get_saved_model_spec(model_path: str) -> ModelSpec:
    inputs = []
    outputs = []
    model = tf.saved_model.load(model_path)
    model_inputs = [
        inp for inp in model.signatures["serving_default"].inputs if "unknown" not in inp.name  # type: ignore
    ]
    model_outputs = [
        out for out in model.signatures["serving_default"].outputs if "unknown" not in out.name  # type: ignore
    ]
    for inp in model_inputs:
        shape = [-1 if x is None else x for x in inp.shape]
        inputs.append(TensorSpec(name=inp.name, shape=shape, dtype=inp.dtype.name))
    for out in model_outputs:
        shape = [-1 if x is None else x for x in out.shape]
        outputs.append(TensorSpec(name=out.name, shape=shape, dtype=out.dtype.name))
    return ModelSpec(inputs=inputs, outputs=outputs)
