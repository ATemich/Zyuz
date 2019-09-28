import threading
from typing import Union
from collections import namedtuple
from dlib import correlation_tracker

Vector = namedtuple('Vector', ['x', 'y'])

class Face:
    def __init__(self, id: str, start: Union[Vector, tuple],
                 size: Union[Vector, tuple], dist: float,
                 tracker: correlation_tracker = None):
        self.id = id
        self.start = start
        if type(self.start) is tuple:
            self.start = Vector(*self.start)
        self.size = size
        if type(self.size) is tuple:
            self.size = Vector(*self.size)
        self.dist = dist
        self.tracker = tracker
        self.recognizing_thread = None

    @property
    def end(self):
        return Vector(self.start.x+self.size.x, self.start.y+self.size.y)

    def __contains__(self, other):
        '''True if other's center is in self rectangle'''
        x = other.start.x + other.size.x/2
        y = other.start.y + other.size.y/2
        return self.start.x <= x <= self.end.x and self.start.y <= y <= self.end.y

    def get_image(self, frame):
        return frame[max(self.start.y, 0):self.end.y, max(self.start.x, 0):self.end.x]

    def update_tracker(self, frame, threshold=7):
        if self.tracker is None:
            return
        if self.tracker.update(frame) < threshold:
            self.tracker = None
            return
        else:
            pos = self.tracker.get_position()
            self.start = Vector(int(pos.left()), int(pos.top()))
            self.size = Vector(int(pos.width()), int(pos.height()))
            return self.tracker

    def get_recognized(self, frame, recognizer, in_background=True):
        if in_background and (self.recognizing_thread is None or not self.recognizing_thread.isAlive()):
            self.recognizing_thread = threading.Thread(target=recognizer.recognize, args=(frame, self))
            self.recognizing_thread.start()
        elif not in_background:
            recognizer.recognize(frame, self)
