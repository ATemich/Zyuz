import numpy
from .. import Face

class FaceRecognizer:
    def __init__(self):
        self.database = None

    def load_from_images(self, path):
        raise NotImplementedError

    def load_from_encodings(self, path):
        raise NotImplementedError

    def recognize(self, pic: numpy.ndarray, face: Face = None) -> str:
        raise NotImplementedError

    @staticmethod
    def cut_out(pic: numpy.ndarray, face: Face = None) -> numpy.ndarray:
        if face is None:
            return pic
        else:
            return pic[face.start.y:face.end.y, face.start.x:face.end.x]
