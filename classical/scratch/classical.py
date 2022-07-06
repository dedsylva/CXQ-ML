########## MNIST Classification Problem ##########
# Developing a CNN that can predict which digit the image has
# We will download the keras.dataset, from which he have 28x28 images with:
# 60000 data to train
# 10000 data to test

import os
from model import Mnist, Imagenet, Covid, Malaria 
from database import MNISTDB, IMAGENETDB, COVIDB, MALARIADB
import time
from sklearn.utils import shuffle

if __name__ == '__main__':

  # Constants
  MNIST_SHAPE = (28,28,1)
  IMAGENET_SHAPE = (100,100,4)
  COVID_SHAPE = (100,100,1)
  MALARIA_SHAPE = (100, 100, 1)
  OPT = 'rmsprop' #'adam'
  LOSS = 'categorical_crossentropy'
  METRICS = ['accuracy', 'AUC']
  BATCH = 32
  EPOCHS = 5 

  RANDOM_LAYERS = 1    # Number of random layers
  SIZE = -1 
  SAVE_PATH = "quantum/scratch/quanvolution/" # Data saving folder
  N_WIRES = 4

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



""" MNIST DATASET """
if MNIST:
  print("*** MNIST DATASET CHOSEN ***")

  db = MNISTDB()
  X_train, Y_train, X_test, Y_test = db.get_data()

  print("Building Architecture of Neural Network...")
  NN = Mnist(MNIST_SHAPE, OPT, LOSS, METRICS)

  model = NN.build_model()
  print("- Model Successfully built. ")

  time.sleep(1)
  print("Training Neural Network")
  Results = NN.train(model, X_train, Y_train, epochs=EPOCHS, batch=BATCH)


  print("Neural Network Successfully Trained!")
  time.sleep(1)

  print("Evaluating model ... ")
  loss, acc, auc = model.evaluate(X_test, Y_test)

  time.sleep(1)
  print(f'Accuracy of Neural Network: {acc}')


""" IMAGENET DATASET (ANTS AND BEES) """
if IMAGENET:
  print("*** IMAGENET (ANTS AND BEES) DATASET CHOSEN ***")
  db = IMAGENETDB(SIZE=IMAGENET_SHAPE)
  X_train, Y_train, X_test, Y_test = db.get_data()
  X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
  X_test, Y_test= shuffle(X_test, Y_test, random_state=0)

  print("Building Architecture of Neural Network...")
  IMG = Imagenet(IMAGENET_SHAPE, OPT, LOSS, METRICS)

  model = IMG.build_model()
  print("- Model Successfully built. ")

  time.sleep(1)
  print("Training Neural Network")
  Results = IMG.train(model, X_train, Y_train, epochs=EPOCHS, batch=BATCH)


  print("Neural Network Successfully Trained!")
  time.sleep(1)

  print("Evaluating model ... ")
  loss, acc = model.evaluate(X_test, Y_test)

  time.sleep(1)
  print(f'Accuracy of Neural Network: {acc}')


""" COVID-19 DATASET  """
if COVID:
  print("*** COVID-19 DATASET CHOSEN ***")
  db = COVIDB(SIZE=COVID_SHAPE)
  X_train, Y_train, X_test, Y_test = db.get_data()

  X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
  X_test, Y_test= shuffle(X_test, Y_test, random_state=0)

  print("Building Architecture of Neural Network...")
  CV = Covid(COVID_SHAPE, OPT, LOSS, METRICS)

  model = CV.build_model()
  print("- Model Successfully built. ")

  time.sleep(1)
  print("Training Neural Network")
  Results = CV.train(model, X_train, Y_train, epochs=EPOCHS, batch=BATCH)


  print("Neural Network Successfully Trained!")
  time.sleep(1)

  print("Evaluating model ... ")
  loss, acc, auc = model.evaluate(X_test, Y_test)

  time.sleep(1)
  print(f'Accuracy: {acc}, AUC: {auc}')


""" MALARIA DATASET  """
if MALARIA:
  print("*** MALARIA DATASET CHOSEN ***")
  pt = os.environ.get('PREPROCESS')
  pt = pt.strip() if pt is not None else pt
  prefix = 'malaria'+pt if pt is not None else 'malaria'

  db = MALARIADB(SAVE_PATH, MALARIA_SHAPE, prefix=prefix)
  SIZE = os.environ.get('SIZE')
  if SIZE is None or SIZE.strip() == '-1':
    SIZE = -1
  elif len(SIZE) == 1:
    SIZE = int(SIZE)
  elif len(SIZE.split(',')) == 2:
    SIZE = (int(SIZE.split(',')[0]), int(SIZE.split(',')[1])) # large datasets we do preprocessing in batches
  else:
    raise ValueError(f'SIZE takes up to 2 arguments, but got {SIZE}')

  X_train, Y_train, X_test, Y_test = db.get_data(SIZE)
  print(X_train.shape)

  print("Building Architecture of Neural Network...")
  M = Malaria(MALARIA_SHAPE, OPT, LOSS, METRICS)

  model = M.build_model()
  print("- Model Successfully built. ")

  time.sleep(1)
  print("Training Neural Network")
  Results = M.train(model, X_train, Y_train, epochs=EPOCHS, batch=BATCH)

  print("Neural Network Successfully Trained!")
  time.sleep(1)

  print("Evaluating model ... ")
  loss, acc, auc = model.evaluate(X_test, Y_test)

  time.sleep(1)
  print(f'Accuracy: {acc}, AUC: {auc}')