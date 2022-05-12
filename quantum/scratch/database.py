import os
from keras.datasets import mnist
from keras.utils import to_categorical
import tensorflow as tf
from pennylane import numpy as np

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


  def get_data(self, size):

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


    return train_images, train_labels, test_images, test_labels 
