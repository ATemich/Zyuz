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
    def cut_out(pic: numpy.ndarray, face: Face = None, padding: int = 0) -> numpy.ndarray:
        if face is None:
            return pic
        else:
            return pic[max(face.start.y-padding, 0):face.end.y+padding,
                       max(face.start.x-padding, 0):face.end.x+padding]
