"""
extract_features_train_classifier is used to extract arcface features from training images and
train the SVM classifier for the training face images.

By H.Iman
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import preprocess_data
import os
import math
import pickle
from sklearn.svm import SVC
import face_model
import cv2

def main(args):
    parser = argparse.ArgumentParser(description='face model test')
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='models/model-r100-ii/model,0', help='path to load model.')
    parser.add_argument('--ga-model', default='gender-age/model/model,0', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id') # ------
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining') # ------
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug') # --------
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold') #------
    args1 = parser.parse_args()
    arcface_model = face_model.FaceModel(args1)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            np.random.seed(seed=args.seed)
            embedding_size = 512

            if args.use_split_dataset:
                dataset_tmp = preprocess_data.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class,
                                                    args.nrof_train_images_per_class)
                if (args.mode == 'TRAIN'):
                    dataset = train_set
                elif (args.mode == 'CLASSIFY'):
                    dataset = test_set
            else:
                dataset = preprocess_data.get_dataset(args.data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert (len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset')

            paths, labels = preprocess_data.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                for j in range(nrof_images):
                    frame = cv2.imread(paths[j])
                    emb_array1, single_face_locations = arcface_model.get_input(frame, arcface_model)
                    emb_array[j] = np.asarray(emb_array1)
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            if (args.mode == 'TRAIN'):
                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)

                # Create a list of class names
                class_names = [cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                # print('Saved classifier model to file "%s"' % classifier_filename_exp)

            elif (args.mode == 'CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)

def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths) >= min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(preprocess_data.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(preprocess_data.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help='Indicates if a new classifier should be trained or a classification ' +
                             'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset',
                        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
                             'Otherwise a separate test set can be specified using the test_data_dir option.',
                        action='store_true')
    parser.add_argument('--test_data_dir', type=str,
                        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
                        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
                        help='Use this number of images from each class for training and the rest for testing',
                        default=10)
    return parser.parse_args(argv)