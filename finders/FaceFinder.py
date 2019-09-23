import numpy
from typing import List, Union
from collections import namedtuple

Vector = namedtuple('Vector', ['x', 'y'])

class Face:
    def __init__(self, id: str, start: Union[Vector, tuple],
                 size: Union[Vector, tuple], dist: float):
        self.id = id
        self.start = start
        if type(self.start) is tuple:
            self.start = Vector(*self.start)
        self.size = size
        if type(self.size) is tuple:
            self.size = Vector(*self.size)
        self.dist = dist
        self.end = Vector(self.start.x+self.size.x, self.start.y+self.size.y)


class FaceFinder:
    def __init__(self):
        pass

    def find(self, pic: numpy.ndarray) -> List[Zyuz.Face]:
        raise NotImplementedError
