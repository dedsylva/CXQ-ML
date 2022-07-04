import math
import os
import matplotlib.pyplot as plt
import numpy as np
from quantum.scratch.preprocess import prep_data
from quantum.scratch.pure import quanv
import tensorflow as tf
from PIL import Image
from pathlib import Path

def multiplot(data, title, show=True):
  n_channels = data.shape[2]

  if n_channels > 1:
    # Number of grids to plot
    grids = math.ceil(np.sqrt(n_channels))
    fig, axes = plt.subplots(grids, grids)
    fig.suptitle(title)

    for i, ax in enumerate(axes.flat):
      if i < n_channels:
        im = data[:, :, i]
        ax.imshow(im, interpolation='nearest', cmap='seismic')

    if show: plt.show()
  else:
    plt.title(title)
    plt.imshow(data)
    if show: plt.show()


if __name__ == '__main__':
  #Tensorflow not showing annyoing Warnings
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

  MNIST_SHAPE = (28,28,4)
  IMAGENET_SHAPE = (100,100,1)
  COVID_SHAPE = (100,100,1)
  MALARIA_SHAPE = (100, 100, 1)

  SAVE_PATH = "quantum/scratch/quanvolution/Randoms/" # Data saving folder
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

    plt.title('Original - MNIST')
    plt.imshow(X_train[0])
    plt.show()

    Y_train = tf.keras.layers.Conv2D(filters=4, 
                                    kernel_size=(2,2), 
                                    activation='relu', 
                                    input_shape=X_train[0:].shape)(X_train[0].reshape(1, X_train[0].shape[0], X_train[0].shape[1], 1))

    multiplot(Y_train[0], title='MNIST - Classical 2D Convolutional Layer', show=True)
    
    train_images = np.array(train_images[..., tf.newaxis])
    test_images = np.array(test_images[..., tf.newaxis])
    q_train_images, q_test_images = prep_data(train_images, test_images, SAVE_PATH, 'mnist', MNIST_SHAPE, SAVE=False, LIMIT=0)
    multiplot(q_train_images[0], title='MNIST - Quantum Hadamard Circuit Pre-Processing', show=True)

    new_q_image = np.load(SAVE_PATH + "mnist_q_train_images.npy")
    multiplot(new_q_image[0], title='MNIST - Quantum Random Pre-Processing ', show=True)

    res = quanv(X_train[0], n_filters=1, input=True)
    multiplot(res, title='MNIST - PURE Quantum Processing', show=True)

  if IMAGENET:
    a = Path(r'./datasets/hymenoptera_data\train\ants')
    b = Path(r'./datasets/hymenoptera_data\train\bees')
    X_ants = []
    X_bees = []

    count = 0
    for i in a.iterdir():
      im = Image.open(i).resize((IMAGENET_SHAPE[0], IMAGENET_SHAPE[1])).convert('L')
      im = np.array(im).reshape(IMAGENET_SHAPE).astype('float32')/255 
      X_ants.append(im)
      count += 1 

      if count >= 3:
        break

    count = 0
    for i in b.iterdir():
      im = Image.open(i).resize((IMAGENET_SHAPE[0], IMAGENET_SHAPE[1])).convert('L')
      im = np.array(im).reshape(IMAGENET_SHAPE).astype('float32')/255 
      X_bees.append(im)
      count += 1 


      if count >= 3:
        break
 
        
    plt.title('Original - ANT')
    plt.imshow(X_ants[0])
    plt.show()

    X_ants = np.array(X_ants)
    X_bees = np.array(X_bees)

    Y_ants = tf.keras.layers.Conv2D(filters=4, 
                                    kernel_size=(2,2), 
                                    activation='relu', 
                                    input_shape=X_ants[0:].shape)(X_ants[0].reshape(1, X_ants[0].shape[0], X_ants[0].shape[1], 1))


    multiplot(Y_ants[0], title='ANT - Classical 2D Convolutional Layer', show=True)
 

    q_train_images, _ = prep_data(X_ants, X_ants, 'NO-SAVE', 'PREFIX', IMAGENET_SHAPE, SAVE=False, LIMIT=0)
    multiplot(q_train_images[0], title='ANT - Quantum Hadamard Circuit Pre-Processing', show=True)

    new_q_image = np.load(SAVE_PATH + "imagenet_q_train_images.npy")
    multiplot(new_q_image[0], title='ANT - Quantum Random Pre-Processing', show=True)

    res = quanv(X_ants[0], n_filters=1, input=True)
    multiplot(res, title='ANT - PURE Quantum Processing', show=True)

    plt.title('Original - BEE')
    plt.imshow(X_bees[0])
    plt.show()


  
    Y_bees = tf.keras.layers.Conv2D(filters=4, 
                                    kernel_size=(2,2), 
                                    activation='relu', 
                                    input_shape=X_bees[0:].shape)(X_bees[0].reshape(1, X_bees[0].shape[0], X_bees[0].shape[1], 1))


    multiplot(Y_bees[0], title='BEE - Classical 2D Convolutional Layer', show=True)
 

    q_train_images, _ = prep_data(X_bees, X_bees, 'NO-SAVE', 'PREFIX', IMAGENET_SHAPE, SAVE=False, LIMIT=0)
    multiplot(q_train_images[0], title='BEE - Quantum Hadamard Circuit Pre-Processing', show=True)

    new_q_image = np.load(SAVE_PATH + "imagenet_q_train_images.npy")
    multiplot(new_q_image[124], title='BEE - Quantum Random Pre-Processing', show=True)

    res = quanv(X_bees[0], n_filters=1, input=True)
    multiplot(res, title='BEE - PURE Quantum Processing', show=True)
 


  if COVID:
    a = Path(r'./datasets/Covid19/train/Covid')
    b = Path(r'./datasets/Covid19/test/Covid')
    X_train = []
    X_test = []

    count = 0
    for i in a.iterdir():
      im = Image.open(i).resize((COVID_SHAPE[0], COVID_SHAPE[1])).convert('L')
      im = np.array(im).reshape(COVID_SHAPE).astype('float32')/255 
      X_train.append(im)
      count += 1 

      if count >= 3:
        break

    count = 0
    for i in b.iterdir():
      im = Image.open(i).resize((COVID_SHAPE[0], COVID_SHAPE[1])).convert('L')
      im = np.array(im).reshape(COVID_SHAPE).astype('float32')/255 
      X_test.append(im)
      count += 1 


      if count >= 3:
        break
 
        
    plt.title('Original - COVID')
    plt.imshow(X_train[0])
    plt.show()

    X_train = np.array(X_train)

    Y_train = tf.keras.layers.Conv2D(filters=4, 
                                    kernel_size=(2,2), 
                                    activation='relu', 
                                    input_shape=X_train[0:].shape)(X_train[0].reshape(1, X_train[0].shape[0], X_train[0].shape[1], 1))


    multiplot(Y_train[0], title='COVID - Classical 2D Convolutional Layer', show=True)
 

    q_train_images, _ = prep_data(X_train, X_test, 'NO-SAVE', 'PREFIX', COVID_SHAPE, SAVE=False, LIMIT=0)
    multiplot(q_train_images[0], title='COVID - Quantum Hadamard Circuit Pre-Processing', show=True)

    new_q_image = np.load(SAVE_PATH + "covid_q_train_images.npy")
    multiplot(new_q_image[0], title='COVID - Quantum Random Pre-Processing', show=True)

    res = quanv(X_train[0], n_filters=1, input=True)
    multiplot(res, title='COVID - PURE Quantum Processing', show=True)

  if MALARIA:
    a = Path(r'./datasets/Malaria/Parasitized')
    b = Path(r'./datasets/Malaria/Uninfected')
    X_train = []
    X_test = []

    count = 0
    for i in a.iterdir():
      im = Image.open(i).resize((MALARIA_SHAPE[0], MALARIA_SHAPE[1])).convert('L')
      im = np.array(im).reshape(MALARIA_SHAPE).astype('float32')/255 
      X_train.append(im)
      count += 1

      # we could be smarter, but this works, we just need a couple images
      if count >= 3:
        break

    count = 0

    for i in b.iterdir():
      im = Image.open(i).resize((MALARIA_SHAPE[0], MALARIA_SHAPE[1])).convert('L')
      im = np.array(im).reshape(MALARIA_SHAPE).astype('float32')/255 
      X_test.append(im)
      count += 1

      if count >= 3:
        break

        
    plt.title('Original - MALARIA')
    plt.imshow(X_train[0])
    plt.show()

    X_train = np.array(X_train)

    Y_train = tf.keras.layers.Conv2D(filters=4, 
                                    kernel_size=(2,2), 
                                    activation='relu', 
                                    input_shape=X_train[0:].shape)(X_train[0].reshape(1, X_train[0].shape[0], X_train[0].shape[1], 1))


    multiplot(Y_train[0], title='MALARIA - Classical 2D Convolutional Layer', show=True)
 
    q_train_images, _ = prep_data(X_train, X_test, 'NO-SAVE', 'PREFIX', MALARIA_SHAPE, SAVE=False, LIMIT=0)
    multiplot(q_train_images[0], title='MALARIA - Quantum Hadamard Circui tPre-Processing', show=True)

    new_q_image = np.load(SAVE_PATH + "malaria1_q_train_images.npy")
    multiplot(new_q_image[0], title='MALARIA - Quantum Random Pre-Processing', show=True)

    res = quanv(X_train[0], n_filters=1, input=True)
    multiplot(res, title='MALARIA - PURE Quantum Processing', show=True)

