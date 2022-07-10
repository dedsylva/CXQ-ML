import os
from pennylane import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

N_WIRES=4
N_OUTPUT=3

# Device for quanvolutional layer
dev = qml.device("default.qubit", wires=N_WIRES)

# Device for prediction layer
# TODO: let the number of wires accordingly to Y.shape[0] (there should be some trick)
dev2 = qml.device("default.qubit", wires=N_OUTPUT)

@qml.qnode(dev)
def circuit(phi, filter):
  # Our quantum filter
  for j in range(len(phi)):
    # TODO: Test with and without parameters
    # qml.RY(params[0]*phi[j], wires=j)
    qml.RY(np.pi*phi[j], wires=j)

  if filter == 1:
    for i in range(len(phi)):
      qml.Hadamard(i)
  elif filter == 2:
    qml.RY(np.pi/2*0.3, wires=1)
  elif filter == 3:
    for i in range(len(phi)):
      qml.Hadamard(i)
      qml.RY(np.pi*0.3, wires=0)
  elif filter == 4:
    for i in range(len(phi)):
      qml.CNOT(wires=[0,1])
      qml.CNOT(wires=[2,3])


  return [qml.expval(qml.PauliZ(j)) for j in range(len(phi))]


@qml.qnode(dev2)
def pred(params, data):
  assert len(data.shape) == 1, f"Data must be 1D for prediction, but got {data.shape}. Try using flatten first"

  # This is the most important loop
  for j in range(len(data)):
    for i in range(N_OUTPUT):
      qml.RY(data[j], wires=i)

  for i in range(N_OUTPUT):
    qml.RY(params[i, 0], wires=i)

  # for _ in range(N_LAYERS):
  #   for i in range(N_OUTPUT):
  #     qml.Hadamard(i)

  # qml.CZ(wires=[1, 0])
  # qml.CZ(wires=[1, 2])
  # qml.CZ(wires=[0, 2])

  for i in range(N_OUTPUT):
    qml.RY(params[i, 1], wires=i)

  if CATEGORIC:
    return [qml.expval(qml.PauliZ(j)) for j in range(N_OUTPUT)]
  else:
    return [qml.expval(qml.PauliZ(j)) for j in range(1)]

def normalize(data):
  return np.array([abs(d) / sum(abs(data)) for d in data], requires_grad=False)

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
        for c in range(n_filters):
          out[j // 2, k // 2, c] = np.sum(circuit([
              im[j, k, 0],
              im[j, k + 1, 0],
              im[j + 1, k, 0],
              im[j + 1, k + 1, 0]
            ], c))

  else:
    for j in range(0, im.shape[0], 2):
      for k in range(0, im.shape[1], 2):
        for c in range(n_filters):
          out[j // 2, k // 2, c] = np.sum(circuit([
                im[j, k, c],
                im[j, k + 1, c],
                im[j + 1, k, c],
                im[j + 1, k + 1, c]
              ], c))

    
  return out

def maxPool(im):
  """ Max Pool with a 2 by 2 kernel(filter) and a stride equal to 2 
      If Input has size (m x n x c) , the output will be (m/2 x n/2 x c) """

  assert im.shape[0] == im.shape[1], f"Quadratic Matrix Only! Expected ({im.shape[0]},{im.shape[0]}) matrix, but got ({im.shape[0]},{im.shape[1]})"
  assert im.shape[0] %2 == 0, f"MaxPool needs shapes to be even, got {im.shape[0]}"
  out = np.zeros((im.shape[0]//2, im.shape[1]//2, im.shape[2]))

  for k in range(0, im.shape[2], 2):
    for i in range(0, im.shape[0], 2):
      for j in range(0, im.shape[1], 2):
        out[i//2,j//2,k] = np.max([
            im[i, j, k],
            im[i, j + 1, k],
            im[i + 1, j, k],
            im[i + 1, j + 1, k]
          ])
 
  return out

def avgPool(im):
  """ Avg Pool with a 2 by 2 kernel(filter) and a stride equal to 2 
      If Input has size (m x n x c) , the output will be (m/2 x n/2 x c) """

  assert im.shape[0] == im.shape[1], f"Quadratic Matrix Only! Expected ({im.shape[0]},{im.shape[0]}) matrix, but got ({im.shape[0]},{im.shape[1]})"
  assert im.shape[0] %2 == 0, f"AvgPool needs shapes to be even, got {im.shape[0]}"
  out = np.zeros((im.shape[0]//2, im.shape[1]//2, im.shape[2]))

  for k in range(0, im.shape[2], 2):
    for i in range(0, im.shape[0], 2):
      for j in range(0, im.shape[1], 2):
        out[i//2,j//2,k] = np.mean([
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
    return - sum([Y[v]*log(guess[i]) + (1-Y[v])* log( 1 - guess[i])  for i in range(len(guess))]) / len(guess)

def cost(params, res):
  guess = normalize(pred(params, res)) if CATEGORIC else np.abs(pred(params, res))
  return loss(guess)


def train(x, y, epochs, batch, show_summary=False):
  # opt = qml.GradientDescentOptimizer(stepsize=0.3)
  opt = qml.AdamOptimizer(stepsize=0.2)
  params = np.ones((3,2),requires_grad=True) # initial params for circuit optimization
  losses = []
  ret = []
  global v, Y 
  Y = y

  if show_summary:
    print(qml.draw(circuit)([1., 2., 3., 4.], n=4))  
    print("\n\n")
    print(qml.draw(pred)(params, np.random.random((N_OUTPUT,)))) 


  if DEBUG:
    print("x      :", x.shape)
    print("y      :", y.shape)
    print("params :", params)
  for i in range(epochs):
    aux = 0.
    # TODO: Can we do batches in the quanv ? 
    # Better approach (we don't need to do all the data broo)
    # We don't even need that v for, just replace by below
    # Y = y[batch_index]
    # X = x[batch_index]
    # batch_index = np.random.randint(0, len(y), (batch,))
    # for j,v in enumerate(batch_index):
    for j,v in enumerate(range(len(y))):

      print(f"Training data {v}...", end="\r", flush=True)
      res = quanv(x[v], n_filters=1, input=True)
      if DEBUG and j == 0: 
        print("quanv  :", res.shape)
      # res = avgPool(res)
      # if DEBUG and j == 0: 
      #   print("avgPool:", res.shape)
      # res = maxPool(res)
      # if DEBUG and j == 0: 
      #   print("maxPool:", res.shape)
      res = flatten(res)
      if DEBUG and j == 0: 
        print("flatten:", res.shape)
      guess = normalize(pred(params, res)) if CATEGORIC else np.abs(pred(params, res))
      dd = sum([1 if g >= 1/N_OUTPUT else 0 for g in guess]) if CATEGORIC else 0.
      new_params, _loss = opt.step_and_cost(cost,params,res) 
      params = new_params[0]
      answer = np.argmax(Y[v]) if CATEGORIC else Y[v]
      _class = np.argmax(guess) if CATEGORIC else guess
      aux += _loss
      # params = np.clip(params, -1,1)
      if DEBUG and j == 0: 
        print("predic :", guess.shape)
        print(guess)
        print("dd     :", dd)
        print("class  :", _class)
        print("answer :", answer)
        print("loss   :", _loss)
        print("params :", params)
        print("\n")

    losses.append(aux/len(y))
    ret.append(np.argmax(guess))
    if i%10 == 0:
      print(f"#Step  : {i}, Loss   : {aux/len(y)}")

  return res, losses, params

def evaluate(X_test, Y_test, params):
  rights = 0
  covid, normal, pneumonia = 0., 0., 0.
 
  for j in range(len(Y_test)):
    res = quanv(X_test[j], n_filters=1, input=True)
    # res = maxPool(res)
    res = flatten(res)
    guess = normalize(pred(params, res)) if CATEGORIC else np.abs(pred(params, res))
    answer = np.argmax(Y_test[j]) if CATEGORIC else Y_test[j]

    if DEBUG:
      print("predicted   :", guess, np.argmax(guess))

    if CATEGORIC:
      _class = np.argmax(guess)
    else:
      _class = 1 if guess >= 0.5 else 0

    if answer == _class: rights += 1

    if DEBUG:
      print("real answer :", Y_test[j], answer)
      print("right       ?", _class == answer)
      print("total rights:", rights)
      print("total tried :", j+1)
      print("\n")

  return rights/len(Y_test) 

def run(X_train, Y_train, X_test, Y_test, layers, batch, categoric, Debug=False, PRINT=False):
  # an ugly little hack for every function be able to call this stuff 
  global DEBUG, N_LAYERS, CATEGORIC 

  DEBUG = Debug
  N_LAYERS = layers 
  CATEGORIC = categoric 

  classes, losses, params = train(X_train[:10], Y_train[:10], epochs=30, batch=batch)
  print(evaluate(X_test, Y_test, params))

  if PRINT:
    import matplotlib.pyplot as plt
    plt.title('Losses during Training')
    plt.plot(losses)
    plt.show()
  