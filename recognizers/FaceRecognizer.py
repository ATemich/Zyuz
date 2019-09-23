import numpy

class FaceRecognizer:
    def __init__(self):
        self.database = None

    def load_from_images(self, path):
        raise NotImplementedError

    def load_from_encodings(self, path):
        raise NotImplementedError

    def recognize(self, pic: numpy.ndarray, face) -> str:
        raise NotImplementedError
