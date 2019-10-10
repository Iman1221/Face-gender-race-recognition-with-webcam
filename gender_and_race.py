
import face_recognition
import pandas as pd
import numpy as np
from gender.pred import *

def gender_race_detector(bb, labels,clf, frame):

    locs = list(bb)
    for l in range(len(locs)):
        locs[l][0] = int(float(locs[l][3])) - int(float(locs[l][1]))
        locs[l][1] = int(float(locs[l][2]))
        locs[l][2] = int(float(locs[l][3]))
        locs[l][3] = int(float(locs[l][0]))

    single_face_locations = []
    for i in range(len(bb)):
        loc0 = int(float(locs[i][3])) - int(float(locs[i][1]))
        loc2 = int(float(bb[i][2]))
        loc3 = int(bb[i][3])
        loc1 = int(bb[i][0])
        loc = (loc0, loc3, loc2, loc1)
        single_face_locations.append(loc)

    face_encodings1 = face_recognition.face_encodings(frame)
    if not face_encodings1:
        return None, None
    pred = pd.DataFrame(clf.predict_proba(face_encodings1),
                        columns = labels)
    pred = pred.loc[:, COLS]
    locs = \
        pd.DataFrame(locs, columns = ['top', 'right', 'bottom', 'left'])
    df = pd.concat([pred, locs], axis=1)

    race_out = []
    gender_out = []
    for row in df.iterrows():
        top, right, bottom, left = row[1][4:].astype(int)
        if row[1]['Male'] >= 0.5:
            gender = 'Male'
        else:
            gender = 'Female'
        gender_out.append(gender)

        race = np.argmax(row[1][1:4])
        race_out.append(race)

    return race_out, gender_out


def predict_gender_of_frames(video_capture,clf, labels):

    aa=0
    try:
        pred, locs = predict_one_image(video_capture, clf, labels)
        locs = \
            pd.DataFrame(locs, columns = ['top', 'right', 'bottom', 'left'])
        df = pd.concat([pred, locs], axis=1)

        # draw
        for row in df.iterrows():
            top, right, bottom, left = row[1][4:].astype(int)
            if row[1]['Male'] >= 0.5:
                gender = 'Male'
            else:
                gender = 'Female'

            race = np.argmax(row[1][1:4])
            text_showed = "{} {}".format(race, gender)

        # img = draw_attributes(img_path, df)
        return race, gender
    except:
        return "White", "Male"