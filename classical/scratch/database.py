from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf
from PIL import Image
from pathlib import Path
import numpy as np

class MNISTDB:
  def __init__(self):
    pass

  def get_data(self):

    #load data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # return train_images, train_labels, test_images, test_labels 

    # Need that channel dimension, normalized float32 tensor
    X_train = train_images.reshape((60000, 28, 28, 1)).astype('float32')/255 
    Y_train =  to_categorical(train_labels)
    X_test = test_images.reshape((10000, 28, 28, 1)).astype('float32')/255 
    Y_test =  to_categorical(test_labels)

    return X_train, Y_train, X_test, Y_test

class IMAGENETDB:
  """
  labels: 1 == ants, 0 == bees
  """
  def __init__(self, SIZE):
    self.SIZE = SIZE

  def get_data(self):
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    ## TRAINING DATA
    a = Path(r'./datasets/hymenoptera_data\train\ants')
    b = Path(r'./datasets/hymenoptera_data\train\bees')

    for i in a.iterdir():

      im = Image.open(i).resize((self.SIZE[0], self.SIZE[1])).convert('L')
      im = np.array(im).reshape(self.SIZE).astype('float32')/255 
      X_train.append(im)
      Y_train.append(1)
    

    for i in b.iterdir():
      im = Image.open(i).resize((self.SIZE[0], self.SIZE[1])).convert('L')
      im = np.array(im).reshape(self.SIZE).astype('float32')/255 
      X_train.append(im)
      Y_train.append(0)


    ## TEST DATA
    a = Path(r'./datasets/hymenoptera_data\val\ants')
    b = Path(r'./datasets/hymenoptera_data\val\bees')

    for i in a.iterdir():
      im = Image.open(i).resize((self.SIZE[0], self.SIZE[1])).convert('L')
      im = np.array(im).reshape(self.SIZE).astype('float32')/255 
      X_test.append(im)
      Y_test.append(1)
    

    for i in b.iterdir():
      im = Image.open(i).resize((self.SIZE[0], self.SIZE[1])).convert('L')
      im = np.array(im).reshape(self.SIZE).astype('float32')/255 
      X_test.append(im)
      Y_test.append(0)
 
 
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)


class COVIDB:
  """
  labels: 0 == covid, 1 == normal, 2 == pneumonia
  """
  def __init__(self, SIZE):
    self.SIZE = SIZE

  def get_data(self):
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    ## TRAINING DATA
    a = Path(r'./datasets/Covid19/train/Covid')
    b = Path(r'./datasets/Covid19/train/Normal')
    c = Path(r'./datasets/Covid19/train/ViralPneumonia')

    for i in a.iterdir():

      im = Image.open(i).resize((self.SIZE[0], self.SIZE[1])).convert('L')
      im = np.array(im).reshape(self.SIZE).astype('float32')/255 
      X_train.append(im)
      Y_train.append(0)
    

    for i in b.iterdir():
      im = Image.open(i).resize((self.SIZE[0], self.SIZE[1])).convert('L')
      im = np.array(im).reshape(self.SIZE).astype('float32')/255 
      X_train.append(im)
      Y_train.append(1)


    for i in c.iterdir():
      im = Image.open(i).resize((self.SIZE[0], self.SIZE[1])).convert('L')
      im = np.array(im).reshape(self.SIZE).astype('float32')/255 
      X_train.append(im)
      Y_train.append(2)


    ## TEST DATA
    a = Path(r'./datasets/Covid19/test/Covid')
    b = Path(r'./datasets/Covid19/test/Normal')
    c = Path(r'./datasets/Covid19/test/ViralPneumonia')

    for i in a.iterdir():
      im = Image.open(i).resize((self.SIZE[0], self.SIZE[1])).convert('L')
      im = np.array(im).reshape(self.SIZE).astype('float32')/255 
      X_test.append(im)
      Y_test.append(0)
    

    for i in b.iterdir():
      im = Image.open(i).resize((self.SIZE[0], self.SIZE[1])).convert('L')
      im = np.array(im).reshape(self.SIZE).astype('float32')/255 
      X_test.append(im)
      Y_test.append(1)
 

    for i in c.iterdir():
      im = Image.open(i).resize((self.SIZE[0], self.SIZE[1])).convert('L')
      im = np.array(im).reshape(self.SIZE).astype('float32')/255 
      X_test.append(im)
      Y_test.append(2)
 
    Y_train =  to_categorical(Y_train)
    Y_test =   to_categorical(Y_test)
 
    return np.array(X_train), Y_train, np.array(X_test), Y_test