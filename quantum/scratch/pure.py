import os
from pennylane import numpy as np
import pennylane as qml

N_WIRES=4

# Device for quanvolutional layer
dev = qml.device("default.qubit", wires=N_WIRES)

# Device for prediction layer
# TODO: let the number of wires accordingly to Y.shape[0] (there should be some trick)
dev2 = qml.device("default.qubit", wires=10)


@qml.qnode(dev)
def circuit(phi, n, input):
  # Encoding of 4 classical input values
  if input:
    for j in range(n):
      qml.RY(np.pi * phi[j], wires=j)

  # Our quantum filter
  # TODO: put filters with parameters, so we can train them later
  for i in range(n):
    qml.Hadamard(i)

  return [qml.expval(qml.PauliZ(j)) for j in range(n)]


@qml.qnode(dev2)
def pred(params):

  for i in range(len(params)):
    qml.RX(params[i], wires=i)
    qml.Hadamard(0)

    qml.Hadamard(1)
    qml.RY(params[i], wires=i)
    qml.Hadamard(1)

  for _ in range(N_LAYERS):
    qml.Hadamard(0)
    qml.Hadamard(2)

    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[2,3])

    qml.Hadamard(1)
    qml.Hadamard(3)

  return [qml.expval(qml.PauliZ(j)) for j in range(10)]

def normalize(data):
  return np.array([d / sum(data) for d in data], requires_grad=False)

def flatten(img): 
  return img.reshape(-1) 


def quanv(im, n_filters, input=True):
  """ Convolution with a 2 by 2 kernel(filter), a stride equal to 2 
      If Input has size (m x n x c) , the output will be (m/2 x n/2 x n_filters) """

  assert im.shape[0] == im.shape[1], f"Quadratic Matrix Only! Expected {im.shape[0]} x {im.shape[0]} matrix, but got {im.shape[0]} x {im.shape[1]}"
  out = np.zeros((im.shape[0]//2, im.shape[1]//2, n_filters))

  if input:
    for j in range(0, im.shape[0], 2):
      for k in range(0, im.shape[1], 2):
        q_results = circuit([
            im[j, k, 0],
            im[j, k + 1, 0],
            im[j + 1, k, 0],
            im[j + 1, k + 1, 0]
          ], n_filters, input=input
        )
        for c in range(n_filters):
          out[j // 2, k // 2, c] = q_results[c]
  else:
    for j in range(0, im.shape[0]//2, 2):
      for k in range(0, im.shape[1]//2, 2):
        for c in range(n_filters):
          q_results = circuit([
              im[j, k, c],
              im[j, k + 1, c],
              im[j + 1, k, c],
              im[j + 1, k + 1, c]
            ], n_filters, input=input
          )
          out[j // 2, k // 2, c] = q_results[c]
    
  return out

def maxPool(im):
  """ Max Pool with a 2 by 2 kernel(filter) and a stride equal to 2 
      If Input has size (m x n x c) , the output will be (m/2 x n/2 x c) """

  assert im.shape[0] == im.shape[1], f"Quadratic Matrix Only! Expected {im.shape[0]} x {im.shape[0]} matrix, but got {im.shape[0]} x {im.shape[1]}"
  out = np.zeros((im.shape[0]//2, im.shape[1]//2, im.shape[2]))

  for k in range(0, im.shape[2]//2, 2):
    for i in range(0, im.shape[0]//2, 2):
      for j in range(0, im.shape[1]//2, 2):
        out[i,j,k] = max([
            im[i, j, k],
            im[i, j + 1, k],
            im[i + 1, j, k],
            im[i + 1, j + 1, k]
          ])
  
  return out

# Dealing with possible zeros
def log(m):
  return np.nan_to_num(np.log2(m + 1e-03))

def loss(guess):
  if CATEGORIC:
    # Categoric Cross - Entropy Loss
    return - sum([Y[v][i]*log(guess[i])  for i in range(len(guess))]) / len(guess)
  else:
    # Binary Cross - Entropy Loss
    return - sum([Y[v][i]*log(guess[i] + (1-Y[v][i])* log( 1 - guess[i]))  for i in range(len(guess))]) / len(guess)

def cost(params):
  ret = normalize(pred(params))
  return loss(ret)


def train(x, y, epochs, batch, show_summary=False):
  opt = qml.AdamOptimizer(stepsize=0.3)
  params = np.array([0.1,0.1,0.1],requires_grad=True) # initial params for circuit optimization
  losses = []
  ret = []
  global v, Y 
  Y = y

  # TODO: add show_summary (output quantum circuit)


  if DEBUG:
    print("x      :", x.shape)
    print("y      :", y.shape)
  for i in range(epochs):
    # TODO: Can we do batches in the quanv ? 
    # Better approach (we don't need to do all the data broo)
    # We don't even need that v for, just replace by below
    # Y = y[batch_index]
    # X = x[batch_index]
    batch_index = np.random.randint(0, len(y), (batch,))
    for v in batch_index:

      print(f"Training data {v}...", end="\r", flush=True)
      res = quanv(x[v], n_filters=4, input=True)
      if DEBUG and v == 0: 
        print("quanv  :", res.shape)
      res = maxPool(res)
      if DEBUG and v == 0: 
        print("maxPool:", res.shape)
      res = flatten(res)
      if DEBUG and v == 0: 
        print("flatten:", res.shape)
      res = normalize(pred(params))
      if DEBUG and v == 0: 
        print("predic :", res.shape)
        print(res)
        print("class  :", np.argmax(res))
        print("answer :", np.argmax(Y[v]))
        print("loss   :", cost(params))
        print("\n")

    params, _loss = opt.step_and_cost(cost,params) 
    losses.append(_loss)
    ret.append(np.argmax(res))
    if i%10 == 0:
      print(f"#Step  : {i+1}, Loss   : {_loss}")

  return res, losses, params

def run(X_train, Y_train, X_test, Y_test, layers, batch, categoric, Debug=False, PRINT=False):
  # an ugly little hack for every function be able to call this stuff 
  global DEBUG, N_LAYERS, CATEGORIC 

  DEBUG = Debug
  N_LAYERS = layers 
  CATEGORIC = categoric 

  classes, losses, params = train(X_train[:100], Y_train[:100], epochs=20, batch=batch)

  if PRINT:
    import matplotlib.pyplot as plt
    plt.title('Losses during Training')
    plt.plot(losses)
    plt.show()
  