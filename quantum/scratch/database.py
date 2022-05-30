import os
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf
from pennylane import numpy as np
from preprocess import prep_data 
from PIL import Image
from pathlib import Path

class MNISTDB:
  def __init__(self, SAVE_PATH, shape, prefix):
    self.SAVE_PATH = SAVE_PATH
    self.shape = shape
    self.prefix= prefix 


  def get_data(self, size, pp):

    #load data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    if size != -1:
      # Reduce dataset size
      train_images = train_images[:size]
      train_labels = train_labels[:size]
      test_images = test_images[:size]
      test_labels = test_labels[:size]

    # Normalize pixel values within 0 and 1
    train_images = train_images / 255
    test_images = test_images / 255

    # Categorical data for labels
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Add extra dimension for convolution channels
    train_images = np.array(train_images[..., tf.newaxis], requires_grad=False)
    test_images = np.array(test_images[..., tf.newaxis], requires_grad=False)


    if pp is not None and pp != '0': 
      prep_data(train_images, test_images, self.SAVE_PATH, self.prefix, self.shape)

    # Load pre-processed images

    q_train_images = np.load(self.SAVE_PATH + f"{self.prefix}_q_train_images.npy")
    q_test_images = np.load(self.SAVE_PATH + f"{self.prefix}_q_test_images.npy")

    return q_train_images, train_labels, q_test_images, test_labels


class IMAGENETDB:
  """
  labels: 1 == ants, 0 == bees
  """
  def __init__(self, SAVE_PATH, shape, prefix):
    self.SAVE_PATH = SAVE_PATH
    self.shape = shape
    self.prefix = prefix 

  def get_data(self, size, pp):
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    ## TRAINING DATA
    a = Path(r'./datasets/hymenoptera_data\train\ants')
    b = Path(r'./datasets/hymenoptera_data\train\bees')

    thresh = 200
    fn = lambda x : 255 if x > thresh else 0
    for i in a.iterdir():

      im = Image.open(i).resize((self.shape[0:2])).convert('L')
      im = np.array(im).reshape((self.shape[0], self.shape[1], 1)).astype('float32')/255 
      X_train.append(im)
      Y_train.append(1)
    

    for i in b.iterdir():
      im = Image.open(i).resize((self.shape[0:2])).convert('L')
      im = np.array(im).reshape((self.shape[0], self.shape[1], 1)).astype('float32')/255 
      X_train.append(im)
      Y_train.append(0)


    ## TEST DATA
    a = Path(r'./datasets/hymenoptera_data\val\ants')
    b = Path(r'./datasets/hymenoptera_data\val\bees')

    for i in a.iterdir():
      im = Image.open(i).resize((self.shape[0:2])).convert('L')
      im = np.array(im).reshape((self.shape[0], self.shape[1], 1)).astype('float32')/255 
      X_test.append(im)
      Y_test.append(1)
    

    for i in b.iterdir():
      im = Image.open(i).resize((self.shape[0:2])).convert('L')
      im = np.array(im).reshape((self.shape[0], self.shape[1], 1)).astype('float32')/255 
      X_test.append(im)
      Y_test.append(0)
 

    if pp is not None and pp != '0': 
      prep_data(np.array(X_train), np.array(X_test), self.SAVE_PATH, self.prefix, self.shape)

    # Load pre-processed images
    q_train_images = np.load(self.SAVE_PATH + f"{self.prefix}_q_train_images.npy")
    q_test_images = np.load(self.SAVE_PATH + f"{self.prefix}_q_test_images.npy")

    return q_train_images, np.array(Y_train), q_test_images, np.array(Y_test)


class COVIDB:
  """
  labels: 0 == covid, 1 == normal, 2 == pneumonia
  """
  def __init__(self, SAVE_PATH, shape, prefix):
    self.SAVE_PATH = SAVE_PATH
    self.shape = shape
    self.prefix = prefix 

  def get_data(self, size, pp):
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    ## TRAINING DATA
    a = Path(r'./datasets/Covid19/train/Covid')
    b = Path(r'./datasets/Covid19/train/Normal')
    c = Path(r'./datasets/Covid19/train/ViralPneumonia')

    for i in a.iterdir():

      im = Image.open(i).resize((self.shape[0], self.shape[1])).convert('L')
      im = np.array(im).reshape(self.shape).astype('float32')/255 
      X_train.append(im)
      Y_train.append(0)
    

    for i in b.iterdir():
      im = Image.open(i).resize((self.shape[0], self.shape[1])).convert('L')
      im = np.array(im).reshape(self.shape).astype('float32')/255 
      X_train.append(im)
      Y_train.append(1)


    for i in c.iterdir():
      im = Image.open(i).resize((self.shape[0], self.shape[1])).convert('L')
      im = np.array(im).reshape(self.shape).astype('float32')/255 
      X_train.append(im)
      Y_train.append(2)


    ## TEST DATA
    a = Path(r'./datasets/Covid19/test/Covid')
    b = Path(r'./datasets/Covid19/test/Normal')
    c = Path(r'./datasets/Covid19/test/ViralPneumonia')

    for i in a.iterdir():
      im = Image.open(i).resize((self.shape[0], self.shape[1])).convert('L')
      im = np.array(im).reshape(self.shape).astype('float32')/255 
      X_test.append(im)
      Y_test.append(0)
    

    for i in b.iterdir():
      im = Image.open(i).resize((self.shape[0], self.shape[1])).convert('L')
      im = np.array(im).reshape(self.shape).astype('float32')/255 
      X_test.append(im)
      Y_test.append(1)
 

    for i in c.iterdir():
      im = Image.open(i).resize((self.shape[0], self.shape[1])).convert('L')
      im = np.array(im).reshape(self.shape).astype('float32')/255 
      X_test.append(im)
      Y_test.append(2)
 
    Y_train =  to_categorical(Y_train)
    Y_test =   to_categorical(Y_test)
 
    if pp is not None and pp != '0': 
      prep_data(np.array(X_train), np.array(X_test), self.SAVE_PATH, self.prefix, self.shape)

    # Load pre-processed images
    q_train_images = np.load(self.SAVE_PATH + f"{self.prefix}_q_train_images.npy")
    q_test_images = np.load(self.SAVE_PATH + f"{self.prefix}_q_test_images.npy")

    return q_train_images, Y_train, q_test_images, Y_test

