import unittest
from model import Model
from database import Database

db = Database()
X_train, Y_train, X_test, Y_test = db.get_data()


class TestMNIST(unittest.TestCase):
	def test_dataset(self):
		assert X_train.shape == (60000, 28, 28, 1)
		assert Y_test.shape == (10000, 28, 28, 1)

	def test_model(self):
		NN = Model((28,28,1), 'adam', 'categorical_crossentropy', ['accuracy'])
		model = NN.build_model()
		Results = NN.train(model, X_train, Y_train, epochs=5, batch=128)
		_, acc = model.evaluate(X_test, Y_test)

		assert acc > 0.95

if __name__ == '__main__':
	unittest.main()		