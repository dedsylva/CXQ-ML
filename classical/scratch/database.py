from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf
import os

class Database():
  def __init__(self, gpus=False):
    if gpus:
      gpus = tf.config.experimental.list_physical_devices('GPU')
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    else:
      pass

  def get_data(self):

    #load data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    return train_images, train_labels, test_images, test_labels 

    # Need that channel dimension, normalized float32 tensor
    X_train = train_images.reshape((60000, 28, 28, 1)).astype('float32')/255 
    Y_train =  to_categorical(train_labels)
    X_test = test_images.reshape((10000, 28, 28, 1)).astype('float32')/255 
    Y_test =  to_categorical(test_labels)

    return X_train, Y_train, X_test, Y_test
