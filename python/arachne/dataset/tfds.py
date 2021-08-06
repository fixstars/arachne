import tensorflow as tf


class DatasetFromTFDS:
    def __init__(self, ds: tf.data.Dataset, image_key: str = "image"):
        self.tfds = ds
        self._image_key = image_key
        self._itr = self.tfds.as_numpy_iterator()

    def __iter__(self):
        return self

    def __next__(self):
        data = self._itr.__next__()
        image = data[self._image_key]
        target = data
        return image, target

    def __len__(self):
        return len(self.tfds)
