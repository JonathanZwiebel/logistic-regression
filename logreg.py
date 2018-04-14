import numpy as np
import matplotlib.pyplot as plt

# Computes the sigmoid of each element in the numpy matrix X
def sigmoid(X):
	s = 1 / (1 + np.exp(-X))
	return s

# Initializes the parameters of the model
def init_parms(features):
	w = np.zeros((features, 1))features
	b = 0
	return w, b

# Computes a forwards and backwards progation 
def propogate(w, b, X, Y):
	m = X.shape[1]
	A = sigmoid(np.dot(w.T, X) + b)
	cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

	dw = (1/m) * np.matmul(X, (A - Y).T)
	db = (1/m) * np.sum(A - Y)

	cost = np.squeeze(cost)

	gradient = {"dw":dw, "db":db}
	return gradient, cost

# Runs a gradient descent learning algorithm for a set number of generations
def learn(w, b, X, Y, learning_rate, generations):
	for i in range(generations):
		gradient, cost = propogate(w, b, X, Y)
		dw = grads["dw"]
		db = grads["db"]

		w = w - learning_rate * dw
		b = b - learning_rate * db
	params = {"w":w, "b":b}
	return params, cost

# Evalute the model on test data
def test_evaluate(w, b, X, Y):
	m = X.shape[1]
	A = sigmoid(np.dot(w.T, X) + b)
	cost = (-1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
	return cost

# Predict values for Y given input data and parameters
def predict(w, b, X):
	m = X.shape[1]
	Y_predict = np.zeros((1, m))
	w = w.reshape(X.shape[0], 1)

	A = sigmoid(np.matmul(w.T, x) + b)
	for i in range(A.shape[1]):
		if A[0, i] > 0.5:
			Y_predict[0, i] = 1
		else:
			Y_predict[0, i] = 0

	return Y_predict

def model(X_train, Y_train, X_test, Y_test, learning_rate, generations, log_rate = 50):
	w, b = init_parms(x_train.shape[1])

	training_costs = []
	tests_costs = []
	for i in range(learning_rate/log_rate):
		params, train_cost = learn(w, b, X_train, Y_train, learning_rate, log_rate)
		w = params["w"]
		b = params["b"]
		test_cost = test_evaluate(w, b, X_test, Y_test)

		print("training cost (" + str(log_rate * (i + 1)) + ") generation: " + train_cost)
		print("test cost (" + str(log_rate * (i + 1)) + ") generation: " + test_cost)
		
		training_costs.append(train_cost)
		test_costs.append(test_cost)

	predict_train = predict(w, b, X_train)
	predict_test = predict(w, b, X_test)
	training_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
	test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100

	print("training accuracy: " + str(training_accuracy))
	print("test accuracy: " + str(test_accuracy))

	mod = {"train_costs":train_costs, 
	"test_costs":test_costs,
	"Y_prediction_test":Y_prediction_test,
	"Y_prediction_train":Y_prediction_train,
	"w":w,
	"b":b,
	"learning_rate":learning_rate,
	"num_iterations":num_iterations}