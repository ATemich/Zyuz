import numpy
import math
from .. import Face, Vector

class DistanceEstimator:
    def __init__(self):
        pass

    def estimate(self, frame: numpy.ndarray, face: Face):
        eyes = self.find_eyes(frame, face)
        distance = self.find_distance_between_points(*eyes)
        dist = self.estimate_distance(distance)
        return dist, eyes

    def find_eyes(self, frame: numpy.ndarray, face: Face):
        raise NotImplementedError

    def estimate_distance(self, distance):
        return distance #FIXME

    @staticmethod
    def find_distance_between_points(first: Vector, second: Vector):
        if type(first) is not Vector:
            first = Vector(*first)
        if type(second) is not Vector:
            first = Vector(*second)
        return math.sqrt((second.x-first.x)**2 + (second.y-first.y)**2)
