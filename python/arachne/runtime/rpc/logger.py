import logging


class Logger(object):
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s %(pathname)s:%(lineno)d] %(message)s")
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    my_logger = logging.Logger("arachne.runtime.rpc")
    my_logger.addHandler(stream_handler)

    @staticmethod
    def logger():
        return Logger.my_logger
