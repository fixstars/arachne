import onnxruntime as ort

from ..data import ModelSpec, TensorSpec


def get_onnx_model_spec(model_path: str) -> ModelSpec:
    inputs = []
    outputs = []

    session = ort.InferenceSession(model_path)
    for inp in session.get_inputs():
        dtype = inp.type.replace("tensor(", "").replace(")", "")
        if dtype == "float":
            dtype = "float32"
        elif dtype == "double":
            dtype = "float64"
        inputs.append(TensorSpec(name=inp.name, shape=inp.shape, dtype=dtype))
    for out in session.get_outputs():
        dtype = out.type.replace("tensor(", "").replace(")", "")
        if dtype == "float":
            dtype = "float32"
        elif dtype == "double":
            dtype = "float64"
        outputs.append(TensorSpec(name=out.name, shape=out.shape, dtype=dtype))
    return ModelSpec(inputs=inputs, outputs=outputs)
