from .. import DistanceEstimator
from ... import Vector
import dlib, os, cv2

class dlibEstimator(DistanceEstimator):
    def __init__(self, path=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         'shape_predictor_5_face_landmarks.dat')):
        super().__init__()
        self.path = path
        self.predictor = dlib.shape_predictor(path)

    def find_eyes(self, frame, face):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = self.predictor(gray, dlib.rectangle(*face.start, *face.end))
        points = []
        for l, r in [(0, 1), (2, 3)]:
            lp = shape.part(l)
            rp = shape.part(r)
            x = (lp.x + rp.x) / 2
            y = (lp.y + rp.y) / 2
            points.append(Vector(x, y))
        return points
