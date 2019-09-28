import numpy
from .. import Face

class FaceRecognizer:
    def __init__(self):
        self.database = None

    def load_from_images(self, path):
        raise NotImplementedError

    def load_from_encodings(self, path):
        raise NotImplementedError

    def recognize(self, frame: numpy.ndarray, face: Face = None) -> str:
        raise NotImplementedError
