import os
from model import Mnist, Imagenet, Covid 
from database import MNISTDB, IMAGENETDB, COVIDB 
import time
from sklearn.utils import shuffle

if __name__ == '__main__':

  # Constants
  MNIST_SHAPE = (28,28,4)
  IMAGENET_SHAPE = (100,100,4)
  COVID_SHAPE = (250,250,1)
  OPT = 'rmsprop' #'adam'
  LOSS = 'categorical_crossentropy'
  METRICS = ['accuracy']
  BATCH = 64
  EPOCHS = 5 

  RANDOM_LAYERS = 1    # Number of random layers
  SIZE = -1 
  SAVE_PATH = "quantum/scratch/quanvolution/" # Data saving folder
  N_WIRES = 4


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

  db = MNISTDB(SAVE_PATH, MNIST_SHAPE, prefix='mnist')
  pp = os.environ.get('PREPROCESS').strip()
  SIZE = os.environ.get('SIZE')
  SIZE = -1 if SIZE is None else int(SIZE)

  X_train, Y_train, X_test, Y_test = db.get_data(SIZE, pp)

  print("Building Architecture of Quantum Neural Network...")
  q= Mnist(MNIST_SHAPE, OPT, LOSS, METRICS)

  q_model = q.build_model()
  print("- Model Successfully built. ")


  time.sleep(1)
  print("Training Quantum Neural Network")
  q_history = q.train(q_model, X_train, Y_train, epochs=EPOCHS, batch=BATCH)


  print("Quantum Neural Network Successfully Trained!")
  time.sleep(1)

  print("Evaluating Quantm model ... ")
  q_loss, q_acc = q_model.evaluate(X_test, Y_test)

  print(f'Accuracy of Quantum Neural Network: {q_acc}')


""" IMAGENET DATASET (ANTS AND BEES) """
if IMAGENET:
  print("*** IMAGENET (ANTS AND BEES) DATASET CHOSEN ***")
  db = IMAGENETDB(SAVE_PATH, IMAGENET_SHAPE, prefix='imagenet')
  pp = os.environ.get('PREPROCESS').strip()
  SIZE = os.environ.get('SIZE')
  SIZE = -1 if SIZE is None else int(SIZE)
  X_train, Y_train, X_test, Y_test = db.get_data(SIZE, pp)
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

  db = COVIDB(SAVE_PATH, COVID_SHAPE, prefix='covid')
  pp = os.environ.get('PREPROCESS').strip()
  SIZE = os.environ.get('SIZE')
  SIZE = -1 if SIZE is None else int(SIZE)
  X_train, Y_train, X_test, Y_test = db.get_data(SIZE, pp)

  exit(0)
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


