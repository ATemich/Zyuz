from .FaceFinder import FaceFinder, Face
import cv2
import numpy

class HaarFaceFinder(FaceFinder):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.cascade = cv2.CascadeClassifier(path)

    def find(self, pic: numpy.ndarray, *args, **kwargs):
        gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        matches = []
        for x, y, w, h in self.cascade.detectMultiScale(gray, *args, **kwargs):
            matches.append(Face(None, (x, y), (w, h), None))
        return matches
