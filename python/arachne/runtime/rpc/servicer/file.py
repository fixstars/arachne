import os
import shutil
import tempfile

from arachne.logger import Logger
from arachne.runtime.rpc.protobuf import fileserver_pb2, fileserver_pb2_grpc

logger = Logger.logger()


def save_chunks_to_file(streams):
    filename = None
    f = None
    for stream in streams:
        if f is None:
            filename = stream.filename
            assert filename
            f = open(filename, "wb")
            continue
        f.write(stream.chunk.buffer)
    if f is not None:
        f.close()
    return filename


class FileServicer(fileserver_pb2_grpc.FileServiceServicer):
    def __init__(self):
        pass

    def make_tmpdir(self, request, context):
        dirname = tempfile.mkdtemp()
        logger.info("create dirname:" + dirname)
        return fileserver_pb2.MakeTmpDirResponse(dirname=dirname)

    def delete_tmpdir(self, request, context):
        dirname = request.dirname
        if os.path.exists(dirname):
            logger.info("delete dirname: " + dirname)
            shutil.rmtree(dirname)
        return fileserver_pb2.DeleteTmpDirResponse()

    def upload(self, request_iterator, context):
        filename = save_chunks_to_file(request_iterator)
        return fileserver_pb2.UploadResponse(filepath=filename)
