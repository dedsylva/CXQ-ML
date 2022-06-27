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


class MALARIADB:
  """
  labels: # 1 is parasitized, 0 is uninfected 
  """
  def __init__(self, SAVE_PATH, shape, prefix):
    self.SAVE_PATH = SAVE_PATH
    self.shape = shape
    self.prefix = prefix 

  def get_data(self, size):
    data = []
    if type(size) == int:
      self.size = size 
    elif len(size) == 2:
      self.size = size 
    else:
      self.size = (0,9999999) 

    ## TRAINING DATA
    a = Path(r'./datasets/Malaria/Parasitized')
    b = Path(r'./datasets/Malaria/Uninfected')

    for i,d in enumerate(a.iterdir()):
      if d.is_file() and  str(d).endswith('png'):
        if i < self.size[0]:
          continue
        elif i >= self.size[1]:
          break
        else:
          im = Image.open(d).resize((self.shape[0], self.shape[1])).convert('L')
          im = np.array(im).reshape(self.shape).astype('float32')/255 
          data.append([im,1])

    for i,d in enumerate(b.iterdir()):
      if d.is_file() and  str(d).endswith('png'):
        if i < self.size[0]:
          continue
        elif i >= self.size[1]:
          break
        else:
          im = Image.open(d).resize((self.shape[0], self.shape[1])).convert('L')
          im = np.array(im).reshape(self.shape).astype('float32')/255 
          data.append([im,0])

    #shuffle
    data = np.array(data)
    np.random.shuffle(data)

    ## TEST DATA
    test_size = int(len(data)*0.9)

    X_train = np.stack(data[:test_size,0], axis=0)
    X_test  = np.stack(data[test_size:,0], axis=0) 

    Y_train = np.stack(data[:test_size,1], axis=0)
    Y_test  = np.stack(data[test_size:,1], axis=0)

    del(data)

    return X_train, Y_train, X_test, Y_test