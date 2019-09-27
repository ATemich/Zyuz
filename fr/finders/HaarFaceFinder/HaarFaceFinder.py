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

    def find(self, pic: numpy.ndarray, padding=15, *args, **kwargs):
        gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        matches = []
        kw = self.find_kwargs.copy()
        kw.update(kwargs)
        for x, y, w, h in self.cascade.detectMultiScale(gray, *args, **kw):
            _x = max(x-padding, 0)
            _y = max(y-padding, 0)
            _w = w + 2*padding
            _h = h + 2*padding
            matches.append(Face(None, (_x, _y), (_w, _h), None))
        return matches
