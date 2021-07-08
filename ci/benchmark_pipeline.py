import argparse
import concurrent.futures
import logging
from typing import Optional

import tvm.autotvm.env as autotvm_env

from arachne.benchmark import benchmark
from arachne.device import Device, device_list, get_device
from arachne.evaluate import evaluate
from arachne.experiment import Experiment, experiment_list, get_experiment
from arachne.frontend import Frontend, DarknetFrontend
from arachne.logger import Logger
from arachne.quantizer import Type as QType


def benchmark_pipeline(
    experiment: Experiment,
    device: Device,
    dtype: QType,
    rpc_tracker: Optional[str] = None,
    rpc_key: Optional[str] = None,
    enable_benchmark: bool = True,
    enable_evaluate: bool = False,
    evaluate_sample: Optional[int] = None
):
    experiment.compile(device, dtype)
    if enable_benchmark:
        benchmark(experiment, device, rpc_tracker, rpc_key)

    if enable_evaluate:
        evaluate(experiment, device, evaluate_sample)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=experiment_list())
    parser.add_argument('--device', choices=device_list(), default='host')
    parser.add_argument('--dtype', choices=[qt.value for qt in QType])
    parser.add_argument('--rpc-tracker')
    parser.add_argument('--rpc-key')
    parser.add_argument('--disable-benchmark', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--evaluate-sample', type=int, default=None)

    args = parser.parse_args()

    logging.getLogger("strategy").setLevel(logging.ERROR)
    logging.getLogger("autotvm").setLevel(logging.ERROR)

    experiment = get_experiment(args.experiment)
    device = get_device(args.device)

    dtype = QType(args.dtype) if args.dtype else QType(device.default_dype)
    logger = Logger.logger()
    if isinstance(experiment.frontend, DarknetFrontend):
        logger.warning(
            "currently darknet only support FP32. dtype is changed to FP32")
        dtype = QType.FP32

    benchmark = not args.disable_benchmark

    benchmark_pipeline(
        experiment, device, dtype, args.rpc_tracker, args.rpc_key, benchmark, args.evaluate, args.evaluate_sample)


if __name__ == '__main__':
    main()
