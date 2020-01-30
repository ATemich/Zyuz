from .. import FaceFinder
from ... import Face
import cv2
import numpy

class HaarFaceFinder(FaceFinder):
    def __init__(self, path, **kwargs):
        super().__init__()
        self.path = path
        self.cascade = cv2.CascadeClassifier(path)
        self.find_kwargs = kwargs

    def find(self, frame: numpy.ndarray, *args, **kwargs):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matches = []
        kw = self.find_kwargs.copy()
        kw.update(kwargs)
        for x, y, w, h in self.cascade.detectMultiScale(gray, *args, **kw):
            matches.append(Face(None, (x, y), (w, h), None))
        return matches
