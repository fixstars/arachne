import itertools
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

import docker
import numpy as np
import tvm.driver.tvmc.frontends as tvmcfrontends
from docker.models.containers import ExecResult

from arachne.logger import Logger
from arachne.pipeline.package import (
    DarknetPackage,
    DarknetPackageInfo,
    Package,
    PackageInfo,
    TF1Package,
    TF1PackageInfo,
    TFLitePackage,
    TFLitePackageInfo,
    TorchScriptPackage,
    TorchScriptPackageInfo,
    TVMPackage,
    TVMPackageInfo,
)
from arachne.pipeline.stage.utils import (
    get_make_dataset_from_params,
    get_preprocess_from_params,
    get_target_from_params,
    get_target_host_from_params,
)
from arachne.types import ArachneDataset, QType

from .._registry import register_stage, register_stage_candidate
from ..stage import Parameter, Stage

logger = Logger.logger()


class VitisAICompiler(Stage):
    _container_work_dirname = "work"
    _container_work_parent_path = "/home/vitis-ai-user"
    _container_work_path = f"{_container_work_parent_path}/{_container_work_dirname}"
    _export_model_filename = "tvmcmodel.tar"
    _calib_inputs_filename = "calib_inputs.npz"
    _vai_script_file = "compile_in_vitis_ai.py"

    @staticmethod
    def _save_calib_inputs(dataset, preprocess, input_info, q_samples, output_dir):
        import torch
        import torchvision.transforms.functional

        # currently assume only one input
        data = []
        for image, _ in itertools.islice(dataset, q_samples):
            if not isinstance(image, torch.Tensor):
                image = torchvision.transforms.functional.to_tensor(image)
            _, preprocessed = preprocess(image, input_info).get_by_index(0)
            data.append(preprocessed)

        data = np.array(data)
        input_name, _ = input_info.get_by_index(0)
        kwargs = {input_name: data}
        np.savez(output_dir / VitisAICompiler._calib_inputs_filename, **kwargs)

    @staticmethod
    def _export_model(model_path, input_info, output_dir):
        shape_dict = {key: tensorinfo.shape for key, tensorinfo in input_info.items()}
        tvmcmodel = tvmcfrontends.load_model(model_path, shape_dict=shape_dict)
        tvmcmodel.save(output_dir / VitisAICompiler._export_model_filename)

    @staticmethod
    def _create_docker_container(image_name):
        # use subprocess beacause docker-py does not support gpu option
        import subprocess

        cmd = f"""
        docker run --rm -it -detach \
        -v /dev/shm:/dev/shm -v /opt/xilinx/dsa:/opt/xilinx/dsa -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
        --gpus all \
        -e UID=$(id -u) -e GID=$(id -g) \
        {image_name} \
        bash
        """
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise Exception(f"docker run failed : {result.stderr}")
        container_id = result.stdout.decode("utf-8").strip()
        logger.info(f"Vitis-AI container is running : {container_id}")
        return container_id

    @staticmethod
    def _copy_files_to_docker_container(client, container, output_dir):
        with tempfile.TemporaryDirectory() as dname:
            # scirpt, npz, mod, params
            tarfilename = dname + "put.tar.gz"
            archive = tarfile.open(tarfilename, mode="w:gz")
            put_files = [
                os.path.dirname(__file__) + "/../../../script/" + VitisAICompiler._vai_script_file,
                output_dir / VitisAICompiler._export_model_filename,
                output_dir / VitisAICompiler._calib_inputs_filename,
            ]
            for put_file in put_files:
                put_file = str(put_file)
                archive.add(
                    put_file,
                    arcname=VitisAICompiler._container_work_dirname
                    + "/"
                    + os.path.basename(put_file),
                )
            archive.close()
            archive_stream = open(tarfilename, "rb").read()
            client.api.put_archive(
                container.id, VitisAICompiler._container_work_parent_path, archive_stream
            )

    @staticmethod
    def _copy_files_from_docker_container(client, container, output_name, output_dir):
        with tempfile.TemporaryDirectory() as dname:
            tarfilename = dname + "get.tar"
            f = open(tarfilename, "wb")
            path_in_container = VitisAICompiler._container_work_path + "/" + output_name
            bits, stat = client.api.get_archive(container.id, path_in_container)
            for chunk in bits:
                f.write(chunk)
            f.close()
            with tarfile.open(tarfilename, mode="r") as tf:
                tf.extractall(path=output_dir)

    @staticmethod
    def _get_vitis_ai_command(target, target_host, q_samples, output_name, input_info):
        input_args = {
            "target": target,
            "workdir": VitisAICompiler._container_work_path,
            "input_layer_name": input_info.get_by_index(0)[0],
            "input_model_name": VitisAICompiler._export_model_filename,
            "output_path": VitisAICompiler._container_work_path + "/" + output_name,
        }
        if target_host is not None:
            input_args["target_host"] = target_host

        python_command = "python3 {}/{} {}".format(
            VitisAICompiler._container_work_path,
            VitisAICompiler._vai_script_file,
            " ".join([f"--{k}='{str(v)}'" for k, v in input_args.items()]),
        )
        cmd = " && ".join(
            [
                "source /etc/profile",
                "conda activate vitis-ai-tensorflow",
                # File permissions sent by put_archive are root
                f"sudo chmod 777 {VitisAICompiler._container_work_path}",
                "export PX_QUANT_SIZE=" + str(q_samples),
                python_command,
            ]
        )
        cmd = f'bash -c "{cmd}"'
        logger.info(cmd)
        return cmd

    @staticmethod
    def _compile_in_vitis_ai(
        client, container, target, target_host, q_samples, output_name, input_info
    ):
        # call lower level API because container.exec_run() method
        # does not return exit code with stream = True.
        cmd = VitisAICompiler._get_vitis_ai_command(
            target, target_host, q_samples, output_name, input_info
        )
        exec_instance = client.api.exec_create(container.id, cmd)
        exec_output = client.api.exec_start(exec_instance["Id"], stream=True)
        exec_result = ExecResult(None, exec_output)

        for line in exec_result.output:
            if line is not None:
                logger.info(line.decode("utf-8"))

        exitcode = client.api.exec_inspect(exec_instance["Id"]).get("ExitCode")
        if exitcode != 0:
            raise Exception("Compile in Vitis-AI container failed.")

    @staticmethod
    def get_name() -> str:
        return "vitis_ai_compiler"

    @staticmethod
    def get_output_info(input: PackageInfo, params: Parameter) -> Optional[PackageInfo]:
        params = VitisAICompiler.extract_parameters(params)
        target = params["target"]
        target_host = params["target_host"]
        if target is None:
            return None
        if "vitis-ai" not in target:
            return None
        if not isinstance(
            input, (TFLitePackageInfo, TorchScriptPackageInfo, DarknetPackageInfo, TF1PackageInfo)
        ):
            return None
        if isinstance(input, TFLitePackageInfo) and input.for_edgetpu:
            return None
        if (
            isinstance(input, (TFLitePackageInfo, TorchScriptPackageInfo))
            and input.qtype != QType.FP32
        ):
            return None
        if params["make_dataset"] is None or params["preprocess"] is None:
            return None

        return TVMPackageInfo(target=target, target_host=target_host)

    @staticmethod
    def extract_parameters(params: Parameter) -> Parameter:
        target = get_target_from_params(params)
        target_host = get_target_host_from_params(params)
        samples = int(params.get("qsample", "256"))
        make_dataset, make_dataset_str = get_make_dataset_from_params(params)
        preprocess, preprocess_str = get_preprocess_from_params(params)

        return {
            "target": target,
            "target_host": target_host,
            "qsample": samples,
            "make_dataset": make_dataset,
            "make_dataset_str": make_dataset_str,
            "preprocess": preprocess,
            "preprocess_str": preprocess_str,
        }

    @staticmethod
    def process(input: Package, params: Parameter, output_dir: Path) -> Package:
        params = VitisAICompiler.extract_parameters(params)
        samples = params["qsample"]
        target = params["target"]
        target_host = params["target_host"]
        assert target is not None
        output_name = "tvm_package.tar"
        client = docker.from_env()
        # use container if specified, otherwise run container
        if "VAI_CONTAINER_ID" in os.environ:
            vai_container_id = os.environ["VAI_CONTAINER_ID"]
        else:
            image_name = os.getenv("VAI_CONTAINER_NAME", "arachne/vitis_ai:latest")
            vai_container_id = VitisAICompiler._create_docker_container(image_name)

        make_dataset = params["make_dataset"]
        assert make_dataset is not None
        dataset = make_dataset()
        assert isinstance(dataset, ArachneDataset)
        preprocess = params["preprocess"]
        assert preprocess is not None

        assert isinstance(input, (TFLitePackage, TorchScriptPackage, DarknetPackage, TF1Package))
        if isinstance(input, DarknetPackage):
            input_filename = input.weight_file
        elif isinstance(input, (TFLitePackage, TorchScriptPackage, TF1Package)):
            input_filename = input.model_file

        container = client.containers.get(vai_container_id)
        try:
            # save calibration images to .npz
            VitisAICompiler._save_calib_inputs(
                dataset, preprocess, input.input_info, samples, output_dir
            )
            # export model to IRModule/param dict
            VitisAICompiler._export_model(input.dir / input_filename, input.input_info, output_dir)
            # copy files to docker container
            VitisAICompiler._copy_files_to_docker_container(client, container, output_dir)
            # compile in Vitis-AI container
            VitisAICompiler._compile_in_vitis_ai(
                client, container, target, target_host, samples, output_name, input.input_info
            )
            # copy files from docker container
            VitisAICompiler._copy_files_from_docker_container(
                client, container, output_name, output_dir
            )
        finally:
            if "VAI_CONTAINER_ID" not in os.environ:
                # stop and autoremove the container
                container.stop()
                logger.info(f"Vitis-AI container {container.id} is removed")

        return TVMPackage(
            dir=output_dir,
            input_info=input.input_info,
            output_info=input.output_info,
            target=target,
            target_host=target_host,
            package_file=Path(output_name),
        )


register_stage(VitisAICompiler)

register_stage_candidate(VitisAICompiler)
