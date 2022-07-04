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
def circuit(phi, n):
  # Our quantum filter
  for j in range(len(phi)):
    # TODO: Test with and without parameters
    # qml.RY(params[0]*phi[j], wires=j)
    qml.RY(np.pi*phi[j], wires=j)

  for i in range(n):
    qml.Hadamard(i)

  return [qml.expval(qml.PauliZ(j)) for j in range(len(phi))]
  # return qml.probs(wires=range(N_WIRES))


@qml.qnode(dev2)
def pred(params, data):
  assert len(data.shape) == 1, f"Data must be 1D for prediction, but got {data.shape}. Try using flatten first"

  for j  in range(len(data)):
    for k in range(10):
      for i in range(len(params)):
        qml.RY(np.pi*params[i]*data[j], wires=k)
        qml.Hadamard(k)

  for _ in range(N_LAYERS):

    qml.CNOT(wires=[0,1])
    qml.CNOT(wires=[2,3])
    qml.CNOT(wires=[4,5])
    qml.CNOT(wires=[6,7])
    qml.CNOT(wires=[8,9])


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
  # TODO: more than 1 filters (how to define each filter?)

  if input:
    for j in range(0, im.shape[0], 2):
      for k in range(0, im.shape[1], 2):
        q_results = circuit([
            im[j, k, 0],
            im[j, k + 1, 0],
            im[j + 1, k, 0],
            im[j + 1, k + 1, 0]
          ], n_filters)

        for c in range(n_filters):
          out[j // 2, k // 2, c] = np.max(q_results[0])
  else:
    for j in range(0, im.shape[0], 2):
      for k in range(0, im.shape[1], 2):
        for c in range(n_filters):
          q_results = circuit([
              im[j, k, c],
              im[j, k + 1, c],
              im[j + 1, k, c],
              im[j + 1, k + 1, c]
            ], n_filters)

          out[j // 2, k // 2, c] = np.max(q_results[0])
    
  return out

def maxPool(im):
  """ Max Pool with a 2 by 2 kernel(filter) and a stride equal to 2 
      If Input has size (m x n x c) , the output will be (m/2 x n/2 x c) """

  assert im.shape[0] == im.shape[1], f"Quadratic Matrix Only! Expected ({im.shape[0]},{im.shape[0]}) matrix, but got ({im.shape[0]},{im.shape[1]})"
  assert im.shape[0] %2 == 0, f"Shapes must be even, got {im.shape[0]}"
  out = np.zeros((im.shape[0]//2, im.shape[1]//2, im.shape[2]))

  # for k in range(0, im.shape[2], 2):
  k = 0
  for i in range(0, im.shape[0], 2):
    for j in range(0, im.shape[1], 2):
      out[i//2,j//2,k] = np.max([
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

def cost(params, res):
  guess = normalize(pred(params, res))
  return loss(guess)


def train(x, y, epochs, batch, show_summary=False):
  opt = qml.GradientDescentOptimizer(stepsize=0.3)
  params = np.array([0.1,0.1,0.1],requires_grad=True) # initial params for circuit optimization
  losses = []
  ret = []
  global v, Y 
  Y = y

  if show_summary:
    print(qml.draw(circuit)([1., 2., 3., 4.], n=4))  
    print("\n\n")
    print(qml.draw(pred)(params, np.random.random((10,)))) 


  if DEBUG:
    print("x      :", x.shape)
    print("y      :", y.shape)
    print("params :", params)
  for i in range(epochs):
    # TODO: Can we do batches in the quanv ? 
    # Better approach (we don't need to do all the data broo)
    # We don't even need that v for, just replace by below
    # Y = y[batch_index]
    # X = x[batch_index]
    batch_index = np.random.randint(0, len(y), (batch,))
    # for j,v in enumerate(batch_index):
    for j,v in enumerate(range(len(y))):

      print(f"Training data {v}...", end="\r", flush=True)
      res = quanv(x[v], n_filters=1, input=True)
      if DEBUG and j == 0: 
        print("quanv  :", res.shape)
      res = maxPool(res)
      if DEBUG and j == 0: 
        print("maxPool:", res.shape)
      res = flatten(res)
      if DEBUG and j == 0: 
        print("flatten:", res.shape)
      guess = normalize(pred(params, res))
      new_params, _loss = opt.step_and_cost(cost,params,res) 
      params = new_params[0]
      if DEBUG and j == 0: 
        print("predic :", guess.shape)
        print(guess)
        print("class  :", np.argmax(guess))
        print("answer :", np.argmax(Y[v]))
        print("loss   :", _loss)
        print("params :", params)
        print("\n")

    losses.append(_loss)
    ret.append(np.argmax(guess))
    if i%10 == 0:
      print(f"#Step  : {i+1}, Loss   : {_loss}")

  return res, losses, params

def run(X_train, Y_train, X_test, Y_test, layers, batch, categoric, Debug=False, PRINT=False):
  # an ugly little hack for every function be able to call this stuff 
  global DEBUG, N_LAYERS, CATEGORIC 

  DEBUG = Debug
  N_LAYERS = layers 
  CATEGORIC = categoric 

  classes, losses, params = train(X_train[:20], Y_train[:20], epochs=50, batch=batch)

  if PRINT:
    import matplotlib.pyplot as plt
    plt.title('Losses during Training')
    plt.plot(losses)
    plt.show()
  