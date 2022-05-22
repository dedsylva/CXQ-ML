from pennylane import numpy as np
import tensorflow as tf
import pennylane as qml
from pennylane.templates import RandomLayers
import keras
from keras import layers, models
from tqdm import tqdm


np.random.seed(0)           # Seed for NumPy random number generator
tf.random.set_seed(0)       # Seed for TensorFlow random number generator
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
def quanv(image, shape):
  """Convolves the input image with many applications of the same quantum circuit."""
  out = np.zeros((shape[0]//2, shape[1]//2, 4))

  # Loop over the coordinates of the top-left pixel of 2X2 squares
  for j in range(0, shape[0], 2):
    for k in range(0, shape[0], 2):
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

def prep_data(X_train, X_test, SAVE_PATH, PREFIX, SHAPE):

  q_train_images = []
  print("Quantum pre-processing of train images:")
  for idx, img in enumerate(tqdm(X_train)):
    q_train_images.append(quanv(img, SHAPE))
  q_train_images = np.asarray(q_train_images)

  q_test_images = []
  print("\nQuantum pre-processing of test images:")
  for idx, img in enumerate(tqdm(X_test)):
    q_test_images.append(quanv(img, SHAPE))
  q_test_images = np.asarray(q_test_images)

  # Save pre-processed images
  np.save(SAVE_PATH + f"{PREFIX}_q_train_images.npy", q_train_images)
  np.save(SAVE_PATH + f"{PREFIX}_q_test_images.npy", q_test_images)

