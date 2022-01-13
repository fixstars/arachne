import tempfile

import numpy as np
import pytest
from tvm.contrib.download import download

from arachne.runtime.rpc import TfliteRuntimeClient, create_channel, create_server


@pytest.mark.xfail
def test_prohibit_multiple_client(rpc_port=5051):
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = tmp_dir + "/model.tflite"
        url = "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/mobilenet.tflite"
        download(url, model_path)

        server = create_server("tflite", rpc_port)
        server.start()

        channel = create_channel(port=rpc_port)
        try:
            client1 = TfliteRuntimeClient(channel, model_path)
            # cannot create multiple clients
            client2 = TfliteRuntimeClient(channel, model_path)
        finally:
            channel.close()
            server.stop(0)


def test_conitnue_first_client(rpc_port=5051):
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = tmp_dir + "/model.tflite"
        url = "https://arachne-public-pkgs.s3.ap-northeast-1.amazonaws.com/models/test/mobilenet.tflite"
        download(url, model_path)

        dummy_input = np.array(np.random.random_sample([1, 224, 224, 3]), dtype=np.float32)  # type: ignore

        server = create_server("tflite", rpc_port)
        server.start()

        channel = create_channel(port=rpc_port)
        client1 = TfliteRuntimeClient(channel, model_path)

        try:
            client1.set_input(0, dummy_input)
            # cannot create multiple clients
            client2 = TfliteRuntimeClient(channel, model_path)
        except:
            # client1 can continue to be used
            client1.run()
            rpc_output = client1.get_output(0)
            client1.finalize()
        finally:
            channel.close()
            server.stop(0)
