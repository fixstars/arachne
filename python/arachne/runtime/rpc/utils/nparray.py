import io

import numpy as np


def nparray_piece_generator(np_array: np.ndarray, CHUNK_SIZE=1024 * 1024):
    with io.BytesIO() as f:
        np.save(f, np_array)
        f.seek(0)
        while True:
            piece = f.read(CHUNK_SIZE)
            if len(piece) == 0:
                return
            yield piece


def generator_to_np_array(request_iterator, byte_extract_func):
    f = io.BytesIO()
    with io.BytesIO() as f:
        for request in request_iterator:
            piece = byte_extract_func(request)
            f.write(piece)
        f.seek(0)
        np_array = np.load(f)
    return np_array
