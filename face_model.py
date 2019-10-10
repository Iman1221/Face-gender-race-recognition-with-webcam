from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
#import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
sys.path.append('src/common')
sys.path.append('src')
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
import face_image
import face_preprocess

from mtcnn.mtcnn import MTCNN
mtcnn_detector = MTCNN()


def do_flip(data):
  for idx in xrange(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  # model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (1,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

class FaceModel:
  def __init__(self, args):
    self.args = args
    # ctx = mx.gpu(args.gpu)
    ctx = mx.cpu(0)
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.model = None
    self.ga_model = None
    print(args.model)
    if len(args.model)>0:
      self.model = get_model(ctx, image_size, args.model, 'fc1')
    if len(args.ga_model)>0:
      self.ga_model = get_model(ctx, image_size, args.ga_model, 'fc1')

    self.threshold = args.threshold
    self.det_minsize = 30 # 50
    # self.det_threshold = [0.8,0.85,0.85]
    self.det_threshold = [0.6,0.7,0.8]
    #self.det_factor = 0.9
    self.image_size = image_size
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    if args.det==0:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=50, accurate_landmark = True, threshold=self.det_threshold)
    else:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=50, accurate_landmark = True, threshold=[0.0,0.0,0.2])
    self.detector = detector


  def get_input(self, face_img, arcface_model):
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    if ret is None:
      return None
    bbox, points = ret
    if bbox.shape[0]==0:
      return None

    single_frame_encs1 = []
    bbox_out = []
    for kk in range(len(bbox)):
      bbox1 = bbox[kk,0:4]
      for i in range(len(bbox1)):
        bbox1[i] = int(float(bbox1[i]))

      points1 = points[kk,:].reshape((2,5)).T
      nimg = face_preprocess.preprocess(face_img, bbox1, points1, image_size='112,112')
      nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
      aligned = np.transpose(nimg, (2,0,1))
      features = arcface_model.get_feature(aligned)
      single_frame_encs1.append(features)
      bbox_out.append(bbox1)
    return single_frame_encs1, bbox_out
  # def get_input(self, face_img, arcface_model,faces_from_faceboxes):
  #   # ret = self.detector.detect_face(face_img, det_type = self.args.det)
  #
  #   bbox = faces_from_faceboxes
  #   if bbox.shape[0]==0:
  #     return None
  #
  #   single_frame_encs1 = []
  #   bbox_out = []
  #   for kk in range(len(bbox)):
  #     bbox1 = bbox[kk,0:4]
  #     for i in range(len(bbox1)):
  #       bbox1[i] = int(float(bbox1[i]))
  #
  #     # points1 = points[kk,:].reshape((2,5)).T
  #     nimg = face_preprocess.preprocess(face_img, bbox1, image_size='112,112')
  #     nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
  #     aligned = np.transpose(nimg, (2,0,1))
  #     cv2.imshow('hhhhh',nimg)
  #     cv2.waitKey()
  #     features = arcface_model.get_feature(aligned)
  #     single_frame_encs1.append(features)
  #     bbox_out.append(bbox1)
  #   return single_frame_encs1, bbox_out

  def get_feature(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding

  def get_ga(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.ga_model.forward(db, is_train=False)
    ret = self.ga_model.get_outputs()[0].asnumpy()
    g = ret[:,0:2].flatten()
    gender = np.argmax(g)
    a = ret[:,2:202].reshape( (100,2) )
    a = np.argmax(a, axis=1)
    age = int(sum(a))

    return gender, age

