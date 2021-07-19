"""
This script must be executed in Vitis-AI container.
"""

import argparse
import logging
import os

import numpy as np
import tvm
from tvm.contrib import graph_executor
from tvm.driver import tvmc
from tvm.driver.tvmc.common import parse_target
from tvm.driver.tvmc.model import TVMCModel

logging.basicConfig()
logger = logging.getLogger("pyxir")
# logger.setLevel(logging.INFO)


class VAI_compiler:
    def __init__(self, target, target_host, workdir, input_layer_name, q_samples):
        targets = parse_target(target)
        for t in targets:
            if t["name"] == "vitis-ai":
                self.dpu_target = t["opts"]["dpu"]
        self.target = targets[-1]["raw"]
        self.target_host = target_host

        self.workdir = workdir
        self.input_layer_name = input_layer_name
        self.q_samples = q_samples
        self.export_rt_mod_file = "vitis_ai.rtmod"

    def load_model(self, input_model_name):
        input_model_path = f"{self.workdir}/{input_model_name}"
        return TVMCModel(model_path=input_model_path)

    def load_calib_inputs(self):
        npz_path = f"{self.workdir}/calib_inputs.npz"
        npz = np.load(npz_path, allow_pickle=True)
        if self.input_layer_name not in npz:
            raise KeyError(f"input_layer_name {self.input_layer_name} is not found in {npz_path}")
        return npz[self.input_layer_name]

    def prepare_quantization(self, tvmcmodel):
        target = f"vitis-ai -dpu={self.dpu_target} -export_runtime_module={self.export_rt_mod_file}, llvm "
        # transform end
        tvmcpackage = tvmc.compiler.compile_model(
            tvmcmodel,
            target=target,
            package_path="./hoge.tar",
            dump_code=["relay"],
            target_host="llvm",
            desired_layout=None,
        )
        return tvmcpackage

    def quantization(self, tvmcpackage, calib_inputs):
        # peform quantization, generate 'vitis_ai.rtmod'
        def create_inference_session(libmod, graph_json_str):
            inference_session = graph_executor.create(graph_json_str, libmod, tvm.cpu())
            return inference_session

        graph_json_str = tvmcpackage.graph
        libpath = tvmcpackage.lib_path
        libmod = tvm.runtime.load_module(libpath)

        inference_session = create_inference_session(libmod, graph_json_str)
        print("Quantize on first {} inputs".format(self.q_samples))
        for i in range(self.q_samples):
            inference_session.set_input(self.input_layer_name, calib_inputs[i])
            inference_session.run()

    def export(self, tvmcmodel, output_path):
        target = f"vitis-ai -dpu={self.dpu_target} -load_runtime_module={self.export_rt_mod_file}, {self.target} "

        tvmc.compiler.compile_model(
            tvmcmodel,
            target=target,
            package_path=output_path,
            export_format="tar",
            dump_code=["relay"],
            target_host=self.target_host,
            desired_layout=None,
        )

    def compile(self, input_model_path, output_path):
        calib_inputs = self.load_calib_inputs()
        tvmcmodel = self.load_model(input_model_path)
        tvmcpackage = self.prepare_quantization(tvmcmodel)
        self.quantization(tvmcpackage, calib_inputs)
        # Do not use IRModule change by the first tvmc.compile_model,
        #  and load original IRModule and perform partition again.
        tvmcmodel = self.load_model(input_model_path)
        self.export(tvmcmodel, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--target_host")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--input_model_name", required=True)
    parser.add_argument("--input_layer_name", required=True)
    parser.add_argument("--output_path", required=True)

    if "PX_QUANT_SIZE" not in os.environ:
        raise KeyError("set PX_QUANT_SIZE to environment used in PyXIR.")
    q_samples = int(os.environ["PX_QUANT_SIZE"])
    args = parser.parse_args()

    vai_compiler = VAI_compiler(
        args.target, args.target_host, args.workdir, args.input_layer_name, q_samples
    )
    vai_compiler.compile(args.input_model_name, args.output_path)


if __name__ == "__main__":
    main()
