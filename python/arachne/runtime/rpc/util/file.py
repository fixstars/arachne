from arachne.runtime.rpc.protobuf import fileserver_pb2, stream_data_pb2

CHUNK_SIZE = 1024 * 1024  # 1MB


def get_file_chunks(src_filepath, dst_filepath):
    with open(src_filepath, "rb") as f:
        yield fileserver_pb2.FileInfo(filename=dst_filepath)
        while True:
            piece = f.read(CHUNK_SIZE)
            if len(piece) == 0:
                return
            chunk = stream_data_pb2.Chunk(buffer=piece)
            fileinfo = fileserver_pb2.FileInfo(chunk=chunk)
            yield fileinfo


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
