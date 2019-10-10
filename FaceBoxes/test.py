import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from PIL import Image, ImageDraw
import os
import cv2
import time

import dlib
import face_recognition

from face_detector import FaceDetector


MODEL_PATH = 'model.pb'
face_detector = FaceDetector(MODEL_PATH)


# # Get an image
def draw_boxes_on_image(image, boxes, scores):

    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy, 'RGBA')
    width, height = image.size

    for b, s in zip(boxes, scores):
        ymin, xmin, ymax, xmax = b
        fill = (255, 0, 0, 45)
        outline = 'red'
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            fill=fill, outline=outline
        )
        draw.text((xmin, ymin), text='{:.3f}'.format(s))
    return image_copy

video_capture = cv2.VideoCapture(0) 
while (True):
    ret, frame = video_capture.read()

    # # # image_array = cv2.imread(frame)
    image_array = cv2.resize(frame, (0, 0), fx=1, fy=1)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_array)

    boxes, scores = face_detector(image_array, score_threshold=0.3)
    im = draw_boxes_on_image(Image.fromarray(image_array), boxes, scores)

    # single_face_locations = face_recognition.face_locations(image_array,model = "cnn")
    # print(len(single_face_locations))
    opencvImage = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    # im.show()
    cv2.imshow('fff',opencvImage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break