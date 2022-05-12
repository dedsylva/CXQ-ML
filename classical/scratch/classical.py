########## MNIST Classification Problem ##########
# Developing a CNN that can predict which digit the image has
# We will download the keras.dataset, from which he have 28x28 images with:
# 60000 data to train
# 10000 data to test

from model import Model
from database import Database
import time

# Constants
SHAPE = (28,28,1)
OPT = 'rmsprop' #'adam'
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']
BATCH = 64
EPOCHS = 5 

db = Database()
X_train, Y_train, X_test, Y_test = db.get_data()


print("Building Architecture of Neural Network...")
NN = Model(SHAPE, OPT, LOSS, METRICS)

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
print(f'Printing Accuracy of Neural Network: {acc}')
