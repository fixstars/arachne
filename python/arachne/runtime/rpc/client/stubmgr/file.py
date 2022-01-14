import os
from pathlib import Path

from arachne.runtime.rpc.protobuf import (
    fileserver_pb2,
    fileserver_pb2_grpc,
    stream_data_pb2,
)

CHUNK_SIZE = 1024 * 1024  # 1MB


def get_file_chunks(src_filepath, dst_filepath):
    with open(src_filepath, "rb") as f:
        yield fileserver_pb2.UploadRequest(filename=dst_filepath)
        while True:
            piece = f.read(CHUNK_SIZE)
            if len(piece) == 0:
                return
            chunk = stream_data_pb2.Chunk(buffer=piece)
            fileinfo = fileserver_pb2.UploadRequest(chunk=chunk)
            yield fileinfo


class FileStubManager:
    def __init__(self, channel):
        self.stub = fileserver_pb2_grpc.FileServerStub(channel)
        response = self.stub.make_tmpdir(fileserver_pb2.MakeTmpDirRequest())
        self.tmpdir = Path(response.dirname)

    def __del__(self):
        try:
            self.stub.delete_tmpdir(fileserver_pb2.DeleteTmpDirRequest(dirname=str(self.tmpdir)))
        except:
            pass

    def upload(self, src_file_path: Path):
        dst_file_path = str(self.tmpdir / os.path.basename(src_file_path))
        chunks_generator = get_file_chunks(src_file_path, dst_file_path)
        response = self.stub.upload(chunks_generator)
        return response
