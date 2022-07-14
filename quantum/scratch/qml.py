import os
import time
from model import Mnist, Imagenet, Covid, Malaria 
from database import MNISTDB, IMAGENETDB, COVIDB, MALARIADB 
from sklearn.utils import shuffle
from pure import run

if __name__ == '__main__':

  # Constants
  MNIST_SHAPE = (28,28,4)
  IMAGENET_SHAPE = (100,100,4)
  COVID_SHAPE = (250,250,1)
  MALARIA_SHAPE = (100, 100, 1)
  OPT = 'adam'
  LOSS = 'categorical_crossentropy'
  METRICS = ['accuracy', 'AUC']
  BATCH = 32
  PURE_BATCH = 16
  EPOCHS = 20 

  RANDOM_LAYERS = 1    # Number of random layers
  SIZE = -1 
  SAVE_PATH = "quantum/scratch/quanvolution/" # Data saving folder
  N_WIRES = 4

  AVAILABLE_MODELS = ['MNIST', 'IMAGENET', 'COVID', 'MALARIA']
  AVAILABLE_TYPES = ['PURE', 'MIXED']

  RANDOM = bool(int(os.environ.get('RANDOM').strip())) if os.environ.get('RANDOM') is not None else False
  DEBUG = bool(int(os.environ.get('DEBUG').strip())) if os.environ.get('DEBUG') is not None else False
  PRINT = bool(int(os.environ.get('PRINT').strip())) if os.environ.get('PRINT') is not None else False
  pp = os.environ.get('PREPROCESS').strip() if os.environ.get('PREPROCESS') is not None else '0'

  MODEL = os.environ.get('MODEL').strip()
  TYPE  = os.environ.get('TYPE').strip() if os.environ.get('TYPE') is not None else 'PURE'
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

  db = MNISTDB(SAVE_PATH, MNIST_SHAPE, prefix='mnist')
  SIZE = os.environ.get('SIZE')
  SIZE = -1 if SIZE is None else int(SIZE)

  if RANDOM:
    X_train, Y_train, X_test, Y_test = db.get_random(SIZE, random_path=r'quantum/scratch/quanvolution/Randoms/')
  else:
    X_train, Y_train, X_test, Y_test = db.get_data(SIZE, pp, TYPE)

  if TYPE == 'PURE':
    run(X_train, Y_train, X_test, Y_test, layers=1, batch=PURE_BATCH, categoric=True, Debug=DEBUG, PRINT=PRINT)

  else:

    print("Building Architecture of Neural Network...")
    MN = Mnist(MNIST_SHAPE, OPT, LOSS, METRICS)

    model = MN.build_model()
    print("- Model Successfully built. ")

    time.sleep(1)
    print("Training Neural Network")
    history = MN.train(model, X_train, Y_train, epochs=EPOCHS, batch=BATCH)


    print("Neural Network Successfully Trained!")
    time.sleep(1)

    print("Evaluating model ... ")
    loss, acc, auc = model.evaluate(X_test, Y_test)

    print(f'Accuracy: {acc}, AUC: {auc}')


""" IMAGENET DATASET (ANTS AND BEES) """
if IMAGENET:
  print("*** IMAGENET (ANTS AND BEES) DATASET CHOSEN ***")
  db = IMAGENETDB(SAVE_PATH, IMAGENET_SHAPE, prefix='imagenet')
  SIZE = os.environ.get('SIZE')
  SIZE = -1 if SIZE is None else int(SIZE)

  if RANDOM:
    X_train, Y_train, X_test, Y_test = db.get_random(SIZE, random_path=r'quantum/scratch/quanvolution/Randoms/')
  else:
    X_train, Y_train, X_test, Y_test = db.get_data(SIZE, pp)

  X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
  X_test, Y_test= shuffle(X_test, Y_test, random_state=0)

  if TYPE == 'PURE':
    run(X_train, Y_train, X_test, Y_test, layers=5, batch=PURE_BATCH, categoric=False, Debug=DEBUG, PRINT=PRINT)

  else:

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
    loss, acc, auc = model.evaluate(X_test, Y_test)

    time.sleep(1)
    print(f'Accuracy: {acc}, AUC: {auc}')


""" COVID-19 DATASET  """
if COVID:
  print("*** COVID-19 DATASET CHOSEN ***")

  db = COVIDB(SAVE_PATH, COVID_SHAPE, prefix='covid')
  SIZE = os.environ.get('SIZE')
  SIZE = -1 if SIZE is None else int(SIZE)

  if RANDOM:
    X_train, Y_train, X_test, Y_test = db.get_random(SIZE, random_path=r'quantum/scratch/quanvolution/Randoms/')
  else:
    X_train, Y_train, X_test, Y_test = db.get_data(SIZE, pp)

  X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
  X_test, Y_test= shuffle(X_test, Y_test, random_state=0)

  if TYPE == 'PURE':
    run(X_train, Y_train, X_test, Y_test, layers=5, batch=PURE_BATCH, categoric=True, Debug=DEBUG, PRINT=PRINT)

  else:
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
  pt = os.environ.get('pt').strip()
  prefix = 'malaria'+pt if pt is not None else 'malaria'

  db = MALARIADB(SAVE_PATH, MALARIA_SHAPE, prefix=prefix)
  SIZE = os.environ.get('SIZE').strip()
  if SIZE is None or SIZE == 'ALL':
    SIZE = -1
  elif len(SIZE) == 1:
    SIZE = int(SIZE)
  elif len(SIZE.split(',')) == 2:
    # for large datasets we do preprocessing in batches
    SIZE = (int(SIZE.split(',')[0]), int(SIZE.split(',')[1])) 
  else:
    raise ValueError(f'SIZE takes up to 2 arguments, but got {SIZE}')


  if RANDOM:
    X_train, Y_train, X_test, Y_test = db.get_random(SIZE, random_path=r'quantum/scratch/quanvolution/Randoms/')
    X_train, Y_train = shuffle(X_train, Y_train, random_state=0)
    X_test, Y_test= shuffle(X_test, Y_test, random_state=0)
  else:
    X_train, Y_train, X_test, Y_test = db.get_data(SIZE, pp)

  if TYPE == 'PURE':
    run(X_train, Y_train, X_test, Y_test, layers=1, batch=PURE_BATCH, categoric=False, Debug=DEBUG, PRINT=PRINT)

  else:
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