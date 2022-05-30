########## MNIST Classification Problem ##########
# Developing a CNN that can predict which digit the image has
# We will download the keras.dataset, from which he have 28x28 images with:
# 60000 data to train
# 10000 data to test

import os
from tkinter.tix import IMAGE
from model import Mnist, Imagenet, Covid 
from database import MNISTDB, IMAGENETDB, COVIDB 
import time
from sklearn.utils import shuffle

# Constants
MNIST_SHAPE = (28,28,1)
IMAGENET_SHAPE = (500,500,1)
COVID_SHAPE = (500,500,1)
OPT = 'rmsprop' #'adam'
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']
BATCH = 32  
EPOCHS = 5 

if __name__ == '__main__':

  AVAILABLE_MODELS = ['MNIST, IMAGENET', 'COVID']

  MODEL = os.environ.get('MODEL').strip()
  MNIST, IMAGENET, COVID = False, False, False

  if MODEL is None:
    raise ValueError(f'Model parameter is required and can\'t be {MODEL}')

  if(MODEL == 'MNIST'):
    MNIST=True
  
  elif(MODEL == 'IMAGENET'):
    IMAGENET=True

  elif(MODEL == 'COVID'):
    COVID=True
  
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
  loss, acc = model.evaluate(X_test, Y_test)

  time.sleep(1)
  print(f'Accuracy of Neural Network: {acc}')


""" IMAGENET DATASET (ANTS AND BEES) """
if IMAGENET:
  print("*** IMAGENET (ANTS AND BEES) DATASET CHOSEN ***")
  db = IMAGENETDB(SIZE=IMAGENET_SHAPE)
  X_train, Y_train, X_test, Y_test = db.get_data()
  X_train, Y_train = shuffle(X_train, Y_train, random_state=0)

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
  loss, acc = model.evaluate(X_test, Y_test)

  time.sleep(1)
  print(f'Accuracy of Neural Network: {acc}')

