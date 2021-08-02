import time

from arachne.pipeline.package.package import import_package
from arachne.runtime import runner_init

# ============================================================================================= #

"""Typicaly, you will modify the following CAPITAL variables"""

# Package Path
PACKAGE_PATH = "./examples/testdata/test_x86_cpu_pkg.tar"

# RPC server/tracker hostname
RPC_HOST = None

# RPC key
RPC_KEY = None

# ============================================================================================= #

# Import an exported package
import_start = time.time()
print("Importing... ", end="")

pkg = import_package(PACKAGE_PATH)

import_duration = time.time() - import_start
print("Done! {} sec".format(import_duration))


# Init runtime module
init_start = time.time()
print("Init runtime... ", end="")

module = runner_init(package=pkg, rpc_tracker=RPC_HOST, rpc_key=RPC_KEY, profile=False)

init_duration = time.time() - init_start
print("Done! {} sec".format(init_duration))


# Benchmarking with dummy inputs
benchmark_start = time.time()
print("Benchmarking... ", end="")

res = module.benchmark(10)
print(res)

benchmark_duration = time.time() - benchmark_start
print("Done! {} sec".format(benchmark_duration))
