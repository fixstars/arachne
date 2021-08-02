from arachne.pipeline.package.package import import_package
from arachne.runtime import runner_init

# ============================================================================================= #

"""Typicaly, you will modify the following CAPITAL variables"""

# Package Path
PACKAGE_PATH='./examples/testdata/test_x86_cpu_pkg.tar'

# RPC server/tracker hostname
RPC_HOST = None

# RPC key
RPC_KEY = None

# ============================================================================================= #

pkg = import_package(PACKAGE_PATH)

# Init runtime module
module = runner_init(pkg, rpc_tracker=RPC_HOST, rpc_key=RPC_KEY)

# Benchmarking with dummy inputs

res = module.benchmark(10)
print(res)
