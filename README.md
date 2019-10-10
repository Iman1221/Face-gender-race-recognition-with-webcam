Face detection, alignment, recognition, tracking, and gender recognition, race detection using webcam.

Thanks to David Sandberg for providing a lot of code:

https://github.com/davidsandberg

clone the repository:
git clone 
# -Install all dependencies :

pip install -r requirements.txt

# -Download a pre-trained facenet model.

This part handles the face -> features conversion.

The quickest way is to download this folder (davidsandberg):
https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit
and put the folder under the facenet_model folder.


# -Setup the training data 

For every person's face you want to recognize, please make a folder and put images of photos that include their face (and no one else) in there.
A folder with training photos of talk host Graham Norton and rapper 50 Cent would look like
Cruise
- C1.jpeg
- C2.jpeg

Johnson
- j1.jpg
- j2.jpg


# -Align all the faces and crop the images.

Run the following script to align the face images:

`python align_training_images.py`

Your training_data_aligned folder should now have folders with images inside them. These images have been aligned to make it easier for the facenet network to analyse.
Without this step facenet will not work correctly.

This code uses mtcnn for detecting and algning images.

# -Train the classifier (SVM Classifier)

Run the following script:
`python classifier_train.py`

for gender and emotion classification please refer to:
https://github.com/oarriaga/face_classification


## -Run the main code 

`python main.py`
