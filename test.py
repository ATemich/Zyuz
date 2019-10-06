from fr.finders import HaarFaceFinder
from fr.finders.HaarFaceFinder.cascades import frontalface
from fr.recognizers import FaceNet
from fr.distance_estimators import dlibEstimator
import cv2
import dlib
import time

cv2.namedWindow("preview")
vc = cv2.VideoCapture('video4.mp4')
#vc = cv2.VideoCapture(0)  #webcam

MULTIPLIER = 1
finder = HaarFaceFinder(frontalface, scaleFactor=1.3, minNeighbors=5, maxSize=(120, 120))
recognizer = FaceNet(threshold=0.55)
recognizer.load_from_images('images')
estimator = dlibEstimator()
frame_counter = 0

faces = []
times = []
tm = time.monotonic()
while vc.isOpened():
    _, frame = vc.read()
    frame = frame[350:-50, 400:-400]
    #frame = cv2.resize(frame, (int(frame.shape[1] * MULTIPLIER), int(frame.shape[0] * MULTIPLIER)))

    if frame_counter % 15 == 0:
        new_faces = finder.find(frame)
        for new_face in new_faces:
            for face in faces:
                if new_face in face or face in new_face:
                    break
            else:
                face = new_face.padded(1.1)
                faces.append(face)
                face.tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(*face.start, *face.end)
                face.tracker.start_track(frame, rect)
    else:
        faces = [f for f in faces if f.update_tracker(frame, threshold=5) is not None]

    img = frame
    for face in faces:
        img = cv2.rectangle(img, face.start, face.end, (255, 0, 0), 2)
        for eye in estimator.find_eyes(frame, face):
            cv2.circle(img, (int(eye.x), int(eye.y)), 2, (255, 0, 0), -1)
        if face.id is None:
            cv2.putText(img, 'Recognizing...', tuple(face.start), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
            face.get_recognized(frame, recognizer)
        else:
            dist = round(estimator.estimate(frame, face.padded(0.85)), 2)
            text = f'{face.id}-{dist}'
            color = (0, 0, 255) if dist >= 20 else (255, 0, 0)
            cv2.putText(img, text, tuple(face.start), cv2.FONT_HERSHEY_DUPLEX, 1, color)

    key = cv2.waitKey(1)
    cv2.imshow("preview", img)
    if key == 27:  # exit on ESC
        break

    #подсчет fps
    frame_counter += 1
    t = time.monotonic() - tm
    tm = time.monotonic()
    times.append(t)
    if len(times) > 10:
        times = times[1:]
    #print(len(times)/sum(times))
cv2.destroyWindow("preview")
