# build.py

import os
from pathlib import Path

from setuptools import Extension
from setuptools.command.build_ext import build_ext

ext_modules = [Extension(name="tvm", sources=[])]


class ExtBuilder(build_ext):
    def run(self):
        ext: Extension
        for ext in self.extensions:
            self.build_tvm(ext)
        super().run()

    def build_tvm(self, ext):

        cwd = Path().absolute()
        extdir = Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        # Identify config.cmake for tvm build
        tvm_dir = Path("3rdparty/tvm")
        config_file = os.environ.get('TVM_CMAKE_CONFIG', None)
        if not config_file:
            config_file = tvm_dir / "cmake" / "config.cmake"
        else:
            config_file = Path(config_file)
        assert config_file.exists()

        # Create a build dir
        build_dir = tvm_dir / "build"
        build_dir.mkdir(exist_ok=True)
        build_config_file = build_dir / "config.cmake"

        # Copy config.cmake to build_dir
        with config_file.open() as fp:
            body = fp.read()
            build_config_file.write_text(body)

        # Build tvm
        os.chdir(build_dir)
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(extdir.parent.absolute()),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        self.spawn(["cmake", "-GNinja", ".."] + cmake_args)

        if os.environ.get('BUILD_TVM_RUNTIME_ONLY', 0) == '1':
            self.spawn(["cmake", "--build", ".", "--target", "runtime"])
        else:
            self.spawn(["cmake", "--build", "."])

        os.chdir(cwd)


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    package_dir = {"arachne": "python/arachne", "tvm": "3rdparty/tvm/python/tvm"}

    # For runtime only packages
    packages = ["arachne." + pkg if pkg.startswith("runtime") else pkg for pkg in setup_kwargs['packages']]
    setup_kwargs.update({"package_dir": package_dir})
    setup_kwargs.update({"packages": packages})
    setup_kwargs.update({"ext_modules": ext_modules, "cmdclass": {"build_ext": ExtBuilder}})
