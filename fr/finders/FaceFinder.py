import numpy
from typing import List
from .. import Face

class FaceFinder:
    def __init__(self):
        pass

    def find(self, pic: numpy.ndarray, padding=0) -> List[Face]:
        raise NotImplementedError
