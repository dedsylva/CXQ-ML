import matplotlib.pyplot as plt
from model import QuantumModel 
from database import Database
from pennylane import numpy as np
import tensorflow as tf
import pennylane as qml
from pennylane.templates import RandomLayers
import keras
from keras import layers, models
import time

# Constants
QUANTUM_SHAPE = (14,14,4)
CLASSICAL_SHAPE = (28,28,1)
OPT = 'rmsprop' #'adam'
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']
BATCH = 64
EPOCHS = 5 

RANDOM_LAYERS = 1    # Number of random layers
SIZE = 100
SAVE_PATH = "quanvolution/" # Data saving folder
PREPROCESS = True           # If False, skip quantum processing and load data from SAVE_PATH
N_WIRES = 4


db = Database()
X_train, Y_train, X_test, Y_test = db.get_data(SIZE)

np.random.seed(0)           # Seed for NumPy random number generator
tf.random.set_seed(0)       # Seed for TensorFlow random number generator
SAVE_PATH = "quanvolution/"
N_WIRES=4
RANDOM_LAYERS=1


dev = qml.device("default.qubit", wires=N_WIRES)

# Random circuit parameters
rand_params = np.random.uniform(high=2 * np.pi, size=(RANDOM_LAYERS, 4))

@qml.qnode(dev)
def circuit(phi ):
  # Encoding of 4 classical input values
  for j in range(4):
    qml.RY(np.pi * phi[j], wires=j)

  # Random quantum circuit
  RandomLayers(rand_params, wires=list(range(4)))

  # Measurement producing 4 classical output values
  return [qml.expval(qml.PauliZ(j)) for j in range(4)]

# convolution with a 2×2 kernel and a stride equal to 2
def quanv(image):
  """Convolves the input image with many applications of the same quantum circuit."""
  out = np.zeros((14, 14, 4))

  # Loop over the coordinates of the top-left pixel of 2X2 squares
  for j in range(0, 28, 2):
    for k in range(0, 28, 2):
      # Process a squared 2x2 region of the image with a quantum circuit
      q_results = circuit([
          image[j, k, 0],
          image[j, k + 1, 0],
          image[j + 1, k, 0],
          image[j + 1, k + 1, 0]
        ]
      )
      # Assign expectation values to different channels of the output pixel (j/2, k/2)
      for c in range(4):
        out[j // 2, k // 2, c] = q_results[c]
  return out


# Pre-process Image on Quanvolutional Layer 

q_train_images = []
print("Quantum pre-processing of train images:")
for idx, img in enumerate(X_train):
  print("{}/{}        ".format(idx + 1, SIZE), end="\r")
  q_train_images.append(quanv(img))
q_train_images = np.asarray(q_train_images)

q_test_images = []
print("\nQuantum pre-processing of test images:")
for idx, img in enumerate(X_test):
  print("{}/{}        ".format(idx + 1, SIZE), end="\r")
  q_test_images.append(quanv(img))
q_test_images = np.asarray(q_test_images)

# Save pre-processed images
np.save(SAVE_PATH + "q_train_images.npy", q_train_images)
np.save(SAVE_PATH + "q_test_images.npy", q_test_images)


# Load pre-processed images
q_train_images = np.load(SAVE_PATH + "q_train_images.npy")
q_test_images = np.load(SAVE_PATH + "q_test_images.npy")

# Visualizing the effect of quantum layer
n_samples = 4
n_channels = 4
fig, axes = plt.subplots(1 + n_channels, n_samples, figsize=(10, 10))
for k in range(n_samples):
  axes[0, 0].set_ylabel("Input")
  if k != 0:
    axes[0, k].yaxis.set_visible(False)
  axes[0, k].imshow(X_train[k, :, :, 0], cmap="gray")

  # Plot all output channels
  for c in range(n_channels):
    axes[c + 1, 0].set_ylabel("Output [ch. {}]".format(c))
    if k != 0:
      axes[c, k].yaxis.set_visible(False)
    axes[c + 1, k].imshow(q_train_images[k, :, :, c], cmap="gray")

plt.tight_layout()
plt.show()


# After pre-processing the images with the quanvolutional layer, train in the classical network

print("Building Architecture of Quantum Neural Network...")
q= QuantumModel(QUANTUM_SHAPE, OPT, LOSS, METRICS)

q_model = q.build_model()
print("- Model Successfully built. ")


time.sleep(1)
print("Training Quantum Neural Network")
q_history = q.train(q_model, q_train_images, Y_train, epochs=EPOCHS, batch=BATCH)


print("Quantum Neural Network Successfully Trained!")
time.sleep(1)

print("Evaluating Quantm model ... ")
q_loss, q_acc = q_model.evaluate(q_test_images, Y_test)

time.sleep(1)
print("\n\n\n*****  CLASSICAL MODEL *****\n\n\n")

print("Building Architecture of Classical Neural Network...")
c= QuantumModel(CLASSICAL_SHAPE, OPT, LOSS, METRICS)

c_model = c.build_model()
print("- Model Successfully built. ")

time.sleep(1)
print("Training Classical Neural Network")
c_history = c.train(c_model, X_train, Y_train, epochs=EPOCHS, batch=BATCH)

print("Classical Neural Network Successfully Trained!")
time.sleep(1)

print("Evaluating classical model ... ")
loss, acc = c_model.evaluate(X_test, Y_test)

time.sleep(2)

print("\n\n\n***** ACCURACY TIME *****\n\n\n")
print(f'Printing Accuracy of Quantum Neural Network: {q_acc}')
print(f'Printing Accuracy of Classical Neural Network: {acc}')


# Classical neural network without quantum Layer

import matplotlib.pyplot as plt

plt.style.use("seaborn")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

ax1.plot(q_history.history["accuracy"], "-ob", label="With quantum layer")
ax1.plot(c_history.history["accuracy"], "-og", label="Without quantum layer")
ax1.set_ylabel("Accuracy")
ax1.set_ylim([0, 1])
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(q_history.history["val_loss"], "-ob", label="With quantum layer")
ax2.plot(c_history.history["val_loss"], "-og", label="Without quantum layer")
ax2.set_ylabel("Loss")
ax2.set_ylim(top=2.5)
ax2.set_xlabel("Epoch")
ax2.legend()
plt.tight_layout()
plt.show()