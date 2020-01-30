import numpy
from typing import List
from .. import Face

class FaceFinder:
    def __init__(self):
        pass

    def find(self, frame: numpy.ndarray) -> List[Face]:
        raise NotImplementedError
