"""
train is used to train the SVM classifier for the
training face images.
Note that the face images must be created by running align_training_images.py

By H.Iman
"""
import extract_features_train_classfier

if __name__ == '__main__':
    args = lambda: None
    args.data_dir = 'faces_of_training_images'
    args.seed = None
    args.use_split_dataset = False
    args.mode = 'TRAIN'
    args.batch_size = 460
    args.image_size = 160 #
    args.classifier_filename = 'trained_classifier/classifier.pkl'
    extract_features_train_classfier.main(args)
