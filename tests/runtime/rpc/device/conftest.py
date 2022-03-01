import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--edgetest",
        action="store_true",
        default=False,
        help="run edge tests: rpc server should be launched on the edge device in advance",
    )
    parser.addoption(
        "--tvm_target_device",
        action="store",
        default="jetson-xavier-nx",
        help="Specify the name of the yaml file under config/tvm_target, it will be loaded as TVMConfig for TVM compilation for edge devices.",
    )
    parser.addoption("--rpc_host", action="store")
    parser.addoption("--rpc_port", action="store", type=int, default=5051)


def pytest_configure(config):
    config.addinivalue_line("markers", "edgetest: mark test to run rpc test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--edgetest"):
        return
    edgetest = pytest.mark.skip(reason="need --edgetest option to run")
    for item in items:
        if "edgetest" in item.keywords:
            item.add_marker(edgetest)
