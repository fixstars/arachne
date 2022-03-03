
Introduction: Tool, Pipeline, and Runtime
=========================================

First, we will introduce three key components in Arachne.


Tool
----

In Arachne, a *Tool* represents a model converter or a model compiler.

Arachne supports various famous tools as listed below:

- TVM: a DNN compiler
- TFLite Converter
- TF-TRT
- Torch2ONNX
- Torch2TRT
- ONNX Simplifier
- OpenVINO2Tensorflow
- OpenVINO Model Optimizer

Tools take a DNN model file as their input and output a TAR file containing a converted or compiled DNN model and a metadata file of it. The metadata file includes the dependency information to execute the output model. For example, when compiling a DNN model for the CUDA target by the Arachne's TVM tool, the file includes the CUDA version to check the version consistency at runtime.

Users can use Arachne's Tools with Arachne CLI and Python interface. We detail how to use them in the next tutorial.


Pipeline
--------

If you want to run multiple tools at once, *Pipeline* is what you should use. Arachne supports to execute different tools in one pipeline. We will describe how to use the feature later.


Runtime
-------

For the machine learning development for edge devices such as Jetson devices, it is usual to develop, train, evaluate, and compile DNN models at different host machines that have more CPU & GPU resources. However, to judge the performance of the models on the target devices, we have to run them on the edge devices. For instance, we can naievely copy the model file from the host to the target device, and then run it on the target device.

To accelereate such development cycle, Arachne supports *Runtime* functionality to ease testing a DNN model on target devices. In Arachne, we wrap three types of DNN runtimes (TFLite, ONNXRuntime, and TVM Graph Exectutor) with common interfaces. These wrappers provide RPC to enable users to run the model on remote devices. Once a RPC server is started at a target device, users can remotely test a DNN model from the host machine. This saves many burdensome steps from users. The usage of Arachne Runtime is documented in subsequent tutorials.