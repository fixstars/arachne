# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import msg_response_pb2 as msg__response__pb2
import tfliteruntime_pb2 as tfliteruntime__pb2


class TfliteRuntimeServerStub(object):
    """interface
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Init = channel.unary_unary(
                '/tfliteruntime.TfliteRuntimeServer/Init',
                request_serializer=tfliteruntime__pb2.InitRequest.SerializeToString,
                response_deserializer=msg__response__pb2.MsgResponse.FromString,
                )
        self.SetInput = channel.stream_unary(
                '/tfliteruntime.TfliteRuntimeServer/SetInput',
                request_serializer=tfliteruntime__pb2.SetInputRequest.SerializeToString,
                response_deserializer=msg__response__pb2.MsgResponse.FromString,
                )
        self.Invoke = channel.unary_unary(
                '/tfliteruntime.TfliteRuntimeServer/Invoke',
                request_serializer=tfliteruntime__pb2.InvokeRequest.SerializeToString,
                response_deserializer=msg__response__pb2.MsgResponse.FromString,
                )
        self.GetOutput = channel.unary_stream(
                '/tfliteruntime.TfliteRuntimeServer/GetOutput',
                request_serializer=tfliteruntime__pb2.GetOutputRequest.SerializeToString,
                response_deserializer=tfliteruntime__pb2.GetOutputResponse.FromString,
                )


class TfliteRuntimeServerServicer(object):
    """interface
    """

    def Init(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetInput(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Invoke(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetOutput(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_TfliteRuntimeServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Init': grpc.unary_unary_rpc_method_handler(
                    servicer.Init,
                    request_deserializer=tfliteruntime__pb2.InitRequest.FromString,
                    response_serializer=msg__response__pb2.MsgResponse.SerializeToString,
            ),
            'SetInput': grpc.stream_unary_rpc_method_handler(
                    servicer.SetInput,
                    request_deserializer=tfliteruntime__pb2.SetInputRequest.FromString,
                    response_serializer=msg__response__pb2.MsgResponse.SerializeToString,
            ),
            'Invoke': grpc.unary_unary_rpc_method_handler(
                    servicer.Invoke,
                    request_deserializer=tfliteruntime__pb2.InvokeRequest.FromString,
                    response_serializer=msg__response__pb2.MsgResponse.SerializeToString,
            ),
            'GetOutput': grpc.unary_stream_rpc_method_handler(
                    servicer.GetOutput,
                    request_deserializer=tfliteruntime__pb2.GetOutputRequest.FromString,
                    response_serializer=tfliteruntime__pb2.GetOutputResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'tfliteruntime.TfliteRuntimeServer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class TfliteRuntimeServer(object):
    """interface
    """

    @staticmethod
    def Init(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tfliteruntime.TfliteRuntimeServer/Init',
            tfliteruntime__pb2.InitRequest.SerializeToString,
            msg__response__pb2.MsgResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetInput(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/tfliteruntime.TfliteRuntimeServer/SetInput',
            tfliteruntime__pb2.SetInputRequest.SerializeToString,
            msg__response__pb2.MsgResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Invoke(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/tfliteruntime.TfliteRuntimeServer/Invoke',
            tfliteruntime__pb2.InvokeRequest.SerializeToString,
            msg__response__pb2.MsgResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetOutput(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/tfliteruntime.TfliteRuntimeServer/GetOutput',
            tfliteruntime__pb2.GetOutputRequest.SerializeToString,
            tfliteruntime__pb2.GetOutputResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
