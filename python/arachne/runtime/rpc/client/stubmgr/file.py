import os
from pathlib import Path

import grpc

from arachne.runtime.rpc.protobuf import (
    fileserver_pb2,
    fileserver_pb2_grpc,
    stream_data_pb2,
)

CHUNK_SIZE = 1024 * 1024  # 1MB


def get_file_chunks(src_filepath, dst_filepath):
    """Generate UploadRequest

    Args:
        src_filepath (str): client side filepath to upload.
        dst_filepath (str): server side filepath to upload.

    Yields:
        UploadRequest: contains destination path to upload or data piece bytes.
    """
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
    """FileStubManager upload files to server."""

    def __init__(self, channel: grpc.Channel):
        """

        Create Temporary Directory on the serverside.

        Args:
            channel (grpc.Channel): channel to connect server
        """
        self.stub = fileserver_pb2_grpc.FileServiceStub(channel)
        response = self.stub.make_tmpdir(fileserver_pb2.MakeTmpDirRequest())
        self.tmpdir = Path(response.dirname)

    def __del__(self):
        """Request to delete temporary directory."""
        try:
            self.stub.delete_tmpdir(fileserver_pb2.DeleteTmpDirRequest(dirname=str(self.tmpdir)))
        except:
            pass

    def upload(self, src_file_path: Path):
        """Request to upload file.

        Args:
            src_file_path (Path): filepath to upload.

        Returns:
            UploadResponse:
        """
        dst_file_path = str(self.tmpdir / os.path.basename(src_file_path))
        chunks_generator = get_file_chunks(src_file_path, dst_file_path)
        response = self.stub.upload(chunks_generator)
        return response
