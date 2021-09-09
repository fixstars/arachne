from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Union

import tvm.rpc
from tvm._ffi.runtime_ctypes import Device as TVMDevice

from arachne.runtime.indexed_ordered_dict import IndexedOrderedDict
from arachne.runtime.package import Package


class RuntimeModule(metaclass=ABCMeta):
    """
    An abstract class for arachne runtime modules.
    Each subclass for this class represents a corresponding tvm runtime.
    """

    module: Any
    tvmdev: TVMDevice
    package: Package

    @abstractmethod
    def __init__(self, package: Package, session: tvm.rpc.RPCSession, profile: bool):
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the RuntimeModule name

        Returns
        ----------
        str : The RuntimeModule name
        """
        pass

    @abstractmethod
    def set_inputs(self, inputs: Union[IndexedOrderedDict, List]):
        """Set inputs to the module

        Parameters
        ----------
        inputs : dict of str to NDArray
        """
        pass

    @abstractmethod
    def run(self):
        """Run forward execution of the graph"""
        pass

    @abstractmethod
    def benchmark(self, repeat: int) -> Dict:
        """Benchmarking the module with dummy inputs

        Parameters
        ----------
        repeats : the number of times to repeat the measurement

        Returns
        -------
        Dict : The information of benchmark results
        """
        pass

    @abstractmethod
    def get_outputs(self, output_info: IndexedOrderedDict) -> IndexedOrderedDict:
        """Get outputs

        Parameters
        ----------
        output_info : a dict containing tensor names to be get

        Returns
        -------
        IndexedOrderdDict : a dict of output tensors
        """
        pass

    def get_input_details(self) -> IndexedOrderedDict:
        """Get the input tesonr information

        Returns
        -------
        IndexedOrderdDict : a dict of tensor name (str) to TensorInfo
        """
        return self.package.input_info.copy()

    def get_output_details(self) -> IndexedOrderedDict:
        """Get the output tesonr information

        Returns
        -------
        IndexedOrderdDict : a dict of tensor name (str) to TensorInfo
        """
        return self.package.output_info.copy()
