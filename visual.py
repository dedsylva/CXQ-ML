import math
import os
import matplotlib.pyplot as plt
import numpy as np
from quantum.scratch.preprocess import prep_data
import tensorflow as tf

def multiplot(data, title, show=True):
  n_channels = data.shape[3]

  # Number of grids to plot
  grids = math.ceil(np.sqrt(n_channels))
  fig, axes = plt.subplots(grids, grids)
  fig.suptitle(title)

  for i, ax in enumerate(axes.flat):
    if i < n_channels:
      im = data[0, :, :, i]
      ax.imshow(im, interpolation='nearest', cmap='seismic')

  if show: plt.show()
  fig.clf()


if __name__ == '__main__':
  #Tensorflow not showing annyoing Warnings
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

  MNIST_SHAPE = (28,28,4)
  IMAGENET_SHAPE = (100,100,4)
  COVID_SHAPE = (100,100,1)
  MALARIA_SHAPE = (100, 100, 1)

  SAVE_PATH = "quantum/scratch/quanvolution/" # Data saving folder
  AVAILABLE_MODELS = ['MNIST', 'IMAGENET', 'COVID', 'MALARIA']
  MODEL = os.environ.get('MODEL').strip()
  MNIST, IMAGENET, COVID, MALARIA = False, False, False, False

  if MODEL is None:
    raise ValueError(f'Model parameter is required and can\'t be {MODEL}')
  
  if MODEL not in AVAILABLE_MODELS:
    raise ValueError(f'Wrong model provided. Got {MODEL} but expected {AVAILABLE_MODELS[:]} ')

  if(MODEL == 'MNIST'):
    MNIST=True
  
  elif(MODEL == 'IMAGENET'):
    IMAGENET=True
  
  elif(MODEL == 'COVID'):
    COVID=True

  elif(MODEL == 'MALARIA'):
    MALARIA=True
  
  else:
    raise ValueError(f'Wrong model provided. Got {MODEL} but expected {AVAILABLE_MODELS[:]} ')


  # Load normal Image
  if MNIST:
    from keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    X_train = train_images.reshape((60000, 28, 28, 1)).astype('float32')/255 

    plt.title('Before - MNIST')
    plt.imshow(X_train[0])
    # plt.show()

    Y_train = tf.keras.layers.Conv2D(filters=32, 
                                    kernel_size=(2,2), 
                                    activation='relu', 
                                    input_shape=X_train[0:].shape)(X_train[0].reshape(1, X_train[0].shape[0], X_train[0].shape[1], 1))

    multiplot(Y_train, title='After - MNIST - Classical 2D Convolutional Layer', show=False)
    
    train_images = np.array(train_images[..., tf.newaxis])
    test_images = np.array(test_images[..., tf.newaxis])
    q_train_images, q_test_images = prep_data(train_images, test_images, SAVE_PATH, 'mnist', MNIST_SHAPE, SAVE=False, LIMIT=0)

    multiplot(q_train_images, title='After - MNIST - Quantum Random Pre-Processing', show=False)

    new_q_image = np.load(SAVE_PATH + "mnist_q_train_images.npy")

    multiplot(new_q_image, title='After - MNIST - Quantum Random Pre-Processing Hadamard Circuit', show=True)

