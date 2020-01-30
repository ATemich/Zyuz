import numpy
from .. import Face

class FaceRecognizer:
    def __init__(self):
        self.database = None

    def load_from_images(self, path):
        raise NotImplementedError

    def load_from_encodings(self, path):
        raise NotImplementedError

    def recognize(self, frame: numpy.ndarray, face: Face):
        image = face.get_image(frame)
        id, confidence = self.recognize_image(image)
        if face is not None:
            face.id = id
            return confidence
        else:
            return id

    def recognize_image(self, image: numpy.ndarray):
        raise NotImplementedError
