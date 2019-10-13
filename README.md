Face detection, alignment, recognition, tracking, and gender recognition, race detection with webcam.

Thanks to David Sandberg for providing a lot of code:
https://github.com/davidsandberg


# -Install all dependencies :

pip install -r requirements.txt

## 1. Training- For training a model for face recognition do the following steps:

a) For every person you want to recognize, make a folder ("training_images") and put images of photos that include their face (and no one else) in there.

A folder with training photos of tom cruise and Dwyane Johnson would look like:

Cruise
- C1.jpg
- C2.jpg

Johnson
- j1.jpg
- j2.jpg

Then run "align_training_images.py". This routine will detect faces in all subfolders in "training_images" and puts the into "faces_of_training_images".

b) Download arcface model and put it in "models/" directory.
You need to download ga-model and put it in "gender-age/model/".


c) Run "train.py" to create the model.
This routine will get face images from "faces_of_training_images" and after extracting features, creates the model, and finally puts the model in "trained_classifier".

# 2. Gender and race detection

Download gender and race model from "https://github.com/wondonghyeon/face-classification/" and place it in "gender/face_model.pkl".

Packages:
dlib
face_recognition
NumPy
pandas
scikit-learn

Dataset:
LFWA+ dataset
For more informaton refer to:
https://github.com/wondonghyeon/face-classification/

# 3. Face detection -Real-time Face Detector with High Accuracy

Faceboxes is lightweight and accrate face detection, and its speed is not decrease with increase in the number of the faces.
Download the model from "https://github.com/TropComplique/FaceBoxes-tensorflow" and place it in "models/faceboxes_model.pb"


# 4. Face tracking -kcf tracker
Opencv's lightweight face tracker is used for face tracking.


# 5. Run the code

Plug in your webcam and run "main.py"

<img src="https://raw.githubusercontent.com/Iman1221/Face-gender-race-recognition-with-webcam/master/test_image.png"
height="300">
