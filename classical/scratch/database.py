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
  def __init__(self):
    pass

  def get_data(self):
    X_train, Y_train = [], []
    X_test, Y_test = [], []

    ## TRAINING DATA
    a = Path(r'./datasets/hymenoptera_data\train\ants')
    b = Path(r'./datasets/hymenoptera_data\train\bees')

    thresh = 200
    fn = lambda x : 255 if x > thresh else 0
    for i in a.iterdir():

      # im = Image.open(i).convert('1').resize((28, 28))
      im = Image.open(i).resize((500, 500)).convert('L')
      im = np.array(im).reshape((500, 500, 1)).astype('float32')/255 
      X_train.append(im)
      Y_train.append(1)
    

    for i in b.iterdir():
      # im = Image.open(i).convert('1').resize((28, 28))
      im = Image.open(i).resize((500, 500)).convert('L')
      im = np.array(im).reshape((500, 500, 1)).astype('float32')/255 
      X_train.append(im)
      Y_train.append(0)


    ## TEST DATA
    a = Path(r'./datasets/hymenoptera_data\val\ants')
    b = Path(r'./datasets/hymenoptera_data\val\bees')

    for i in a.iterdir():
      im = Image.open(i).resize((500, 500)).convert('L')
      im = np.array(im).reshape((500, 500, 1)).astype('float32')/255 
      X_test.append(im)
      Y_test.append(1)
    

    for i in b.iterdir():
      im = Image.open(i).resize((500, 500)).convert('L')
      im = np.array(im).reshape((500, 500, 1)).astype('float32')/255 
      X_test.append(im)
      Y_test.append(0)
 
 
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)