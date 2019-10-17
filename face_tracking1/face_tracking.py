import cv2
import datetime
from PIL import Image, ImageDraw
from face_detector import FaceDetector
import numpy as np

MODEL_PATH = 'models/faceboxes_model.pb'
face_detector = FaceDetector(MODEL_PATH)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)

def draw_boxes(frame, boxes,best_name,color,best_class_probabilities,race,gender):
    # global counter
    counter = 0
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
        text_x = x
        text_y = h
        person_name = best_name[counter]
        cv2.putText(frame, race[counter], (int(x), int(h)), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 0, 255), thickness=1, lineType=2)
        cv2.putText(frame, gender[counter], (int(x), int(h) + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 0, 255), thickness=1, lineType=2)
        if (best_class_probabilities[counter] > 0.45):
            cv2.putText(frame, person_name, (int(text_x), int(text_y) + 40), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (0, 0, 255), thickness=1, lineType=2)
        else:
            cv2.putText(frame, 'Unknown!', (int(text_x), int(text_y) + 40), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (0, 0, 255), thickness=1, lineType=2) 
        counter = counter + 1
    return frame

def resize_image(image, size_limit=500.0):
    max_size = max(image.shape[0], image.shape[1])
    if max_size > size_limit:
        scale = size_limit / max_size
        _img = cv2.resize(image, None, fx=scale, fy=scale)
        return _img
    return image

class FaceTracker():
    
    def __init__(self, frame, face):
        (x,y,w,h) = face
        print(x,y,w,h)
        self.face = (x,y,w,h)
        # Arbitrarily picked KCF tracking
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, self.face)
    
    def update(self, frame):
        _, self.face = self.tracker.update(frame)
        print("update")
        return self.face

class Controller():
    
    def __init__(self, event_interval=2):
        self.event_interval = event_interval
        self.last_event = datetime.datetime.now()

    def trigger(self):
        """Return True if should trigger event"""
        return self.get_seconds_since() > self.event_interval
    
    def get_seconds_since(self):
        current = datetime.datetime.now()
        seconds = (current - self.last_event).seconds
        return seconds

    def reset(self):
        self.last_event = datetime.datetime.now()

class Pipeline():
    def __init__(self, event_interval=2):
        self.controller = Controller(event_interval=event_interval)    
        self.detector = face_detector
        self.trackers = []
    
    def detect_and_track(self,frame):
        # get faces 
        image_array = cv2.resize(frame, (0, 0), fx=1, fy=1)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_array)
        faces, scores = face_detector(image_array, score_threshold=0.3)
        faces1 = np.zeros((len(faces), 4))
        for i in range(len(faces)):
            faces1[i][0] = int(faces[i][1])
            faces1[i][1] = int(faces[i][0])
            faces1[i][2] = int(faces[i][2] - faces[i][0])
            faces1[i][3] = int(faces[i][2] - faces[i][0])

        # reset timer
        self.controller.reset()

        # get trackers
        self.trackers = [FaceTracker(frame, face) for face in faces1]

        # if no faces detected, faces will be a tuple.
        new = type(faces1) is not tuple
        return faces1, new
    
    def track(self, frame):
        boxes = [t.update(frame) for t in self.trackers]
        # return state = False for existing boxes only
        return boxes, False
    
    def boxes_for_frame(self, frame):
        if self.controller.trigger():
            return self.detect_and_track(frame)
            
        else:
            return self.track(frame)
