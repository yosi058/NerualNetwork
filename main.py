import numpy
import pandas as pd
import numpy as np
import random as random

# learning rate
LR = 0.001
# Do run noise
NOISE = False
LOAD = False


class NeuralNetwork(object):
    def __init__(self):
        np.random.seed(42)
        # parameters
        self.inputSize = 3072
        self.outputSize = 10
        self.hiddenSize_one = 2500
        self.hiddenSize_two = 1450
        # Do load the weights from files
        if LOAD:
            self.W1 = np.loadtxt("w1.csv", delimiter=',')
            self.W2 = np.loadtxt("w2.csv", delimiter=',')
            self.W3 = np.loadtxt("w3.csv", delimiter=',')
        else:
            # Adding one to the bias
            self.W1 = np.random.uniform(low=-0.01, high=0.01, size=(self.inputSize + 1, self.hiddenSize_one + 1))
            self.W2 = np.random.uniform(low=-0.01, high=0.01, size=(self.hiddenSize_one + 1, self.hiddenSize_two + 1))
            self.W3 = np.random.uniform(low=-0.01, high=0.01, size=(self.hiddenSize_two + 1, self.outputSize))

    # Receiving a vector and returning the output
    def feedForward(self, cur_row):
        # forward propogation through the network
        cur_row = np.array((cur_row), dtype=float)
        # the bias
        cur_row = np.append(cur_row, -1)
        # dot the input with the first matrix
        self.hidden_layer_one = np.dot(cur_row, self.W1)
        # Activation of the activation function
        self.hidden_error_one_sigmoid = self.activationFunction(self.hidden_layer_one)
        # the bias
        self.hidden_error_one_sigmoid[self.hiddenSize_one] = -1
        # Converting a vector to a 1 * 1 matrix
        self.hidden_error_one_sigmoid = np.asmatrix(self.hidden_error_one_sigmoid, dtype=float)
        self.hidden_error_one_sigmoid = np.array(self.hidden_error_one_sigmoid, dtype=float)
        # hidden layer one * w2
        self.hidden_layer_two = np.dot(self.hidden_error_one_sigmoid, self.W2)
        # Activation of the activation function
        self.hidden_layer_tow_sigmoid = self.activationFunction(self.hidden_layer_two[0])
        # the bias
        self.hidden_layer_tow_sigmoid[self.hiddenSize_two] = -1
        # Converting a vector to a 1 * 1 matrix
        self.hidden_layer_tow_sigmoid = np.asmatrix(self.hidden_layer_tow_sigmoid, dtype=float)
        self.hidden_layer_tow_sigmoid = np.array(self.hidden_layer_tow_sigmoid, dtype=float)

        # dot the hidden layer 2 and second set of weights
        self.temp_output = np.dot(self.hidden_layer_tow_sigmoid,
                                  self.W3)
        output = self.activationFunction(self.temp_output[0])
        return output

    # Activation function
    def activationFunction(self, s, deriv=False):
        count = 0
        if (deriv == True):
            for i in s:
                if i > 0:
                    s[count] = 1
                else:
                    s[count] = 0
                count += 1
        else:
            for j in s:
                if j < 0:
                    s[count] = 0
                else:
                    s[count] = min(1, j)
                count += 1
        return s

    # backward propogate through the network
    def backward(self, cur_row, correct_output, output_forward):
        # the bias
        cur_row = np.append(cur_row, -1)
        # Converting a vector to a 1 * 1 matrix
        cur_row = np.asmatrix(cur_row, dtype=float)
        cur_row = np.array(cur_row, dtype=float)
        # Converting a vector to a 1 * 1 matrix
        correct_output = np.asmatrix(correct_output, dtype=float)
        correct_output = np.array(correct_output, dtype=float)
        # Converting a vector to a 1 * 1 matrix
        output_forward = np.asmatrix(output_forward, dtype=float)
        output_forward = np.array(output_forward, dtype=float)

        # error in output 1x10
        self.output_error = correct_output - output_forward
        # sigma(w_ij*error_j)
        self.hidden_two_mult_error = np.dot(self.output_error, self.W3.T)
        # Activation of the activation function
        sigmo = self.activationFunction(self.hidden_layer_two[0], deriv=True)
        # f'(x_i)* func(w_ij*error_j)
        self.hidden_error_two = sigmo * self.hidden_two_mult_error
        # func(w_ij*error_j)
        self.hidden_one_mult_error = np.dot(self.hidden_error_two, self.W2.T)
        # Activation of the activation function
        sigmo_one = self.activationFunction(self.hidden_layer_one, deriv=True)
        # f'(x_i)* sigma(w_ij*error_j)
        self.hidden_error_one = sigmo_one * self.hidden_one_mult_error

        self.W3 += LR * self.hidden_layer_tow_sigmoid.T.dot(self.output_error)
        self.W2 += LR * self.hidden_error_one_sigmoid.T.dot(self.hidden_error_two)
        temp = self.activationFunction(cur_row[0])
        # Converting a vector to a 1 * 1 matrix
        temp = np.asmatrix(temp, dtype=float)
        temp = np.array(temp, dtype=float)
        self.W1 += LR * temp.T.dot(self.hidden_error_one)

    # The function randomly resets 10% of the input
    def noise(self, cur_row):
        # 10%
        range_row = (self.inputSize) * 0.1
        # convert to int
        range_row = int(range_row)
        for p in range(range_row):
            # choose index
            index = np.random.randint(0, self.inputSize - 1)
            cur_row[index] = 0
        return cur_row

    # The function activates the neural network and then learns the error
    def train(self, cur_row, correct_output):
        # Do turn on Noise
        if NOISE:
            cur_row = self.noise(cur_row)
        output_forward = self.feedForward(cur_row)
        self.backward(cur_row, correct_output, output_forward)


# Upload the train file
train_data = pd.read_csv("train.csv", header=None)
# The first column with the answers
first_col = train_data.loc[:, 0]
# Remove the first column
train_data = train_data.drop(0, axis=1)
# Upload the validate file
validate_data = pd.read_csv("test.csv", header=None)
# The first column with the answers
first_col_validate = validate_data.loc[:, 0]
# Remove the first column
validate_data = validate_data.drop(0, axis=1)
# An array that will hold the right results
current_output = []
# Create a neural network (if true the weights would be initialized from files)
NN = NeuralNetwork()
maxi = 0
epoch = 0
percent = 0
counter = 0
output = []
# The amount of epoch
if not LOAD:
    for f in range(60):
        epoch = f
        # At 33 the Noise is turned on and the LR is lowered
        if epoch == 33:
            NOISE = True
            LR = 0.0004
        # Lower the LR
        if epoch == 43:
            LR = 0.00004
        # trains the NN 8000 times
        for i in range(8000):
            # Reset
            current_output = np.zeros(10)
            # The right result
            current_output[first_col[i] - 1] = 1
            # Create a vector of the contemporary line
            current_output = np.array(current_output, dtype=float)
            cur_row = train_data.loc[:i, ]
            cur_row = np.array(cur_row, dtype=float)
            NN.train(cur_row[i], current_output)

        # The current line in validate
        current_output_validate = []
        output_forward = []
        avg = 0
        counter = 0

    # keep the weight to files.
    numpy.savetxt('w1.csv', NN.W1, delimiter=",")
    numpy.savetxt('w2.csv', NN.W1, delimiter=",")
    numpy.savetxt('w3.csv', NN.W1, delimiter=",")
# run the test file anyway.
for t in range(1000):
    # Create a vector of the contemporary line
    cur_row = validate_data.loc[:t, ]
    cur_row = np.array(cur_row, dtype=float)
    output_forward = NN.feedForward(cur_row[t])
    output_forward = np.array(output_forward, dtype=float)
    # Finding the Index of the Largest Organ
    max_index = np.argmax(output_forward, axis=0)
    output.append(max_index + 1)

# keep the test results
output = np.array(output, dtype=int)
# numpy.savetxt('output.txt', output, newline="\n")
numpy.savetxt(fname='output.csv', X=output.astype(int), fmt='%.0f')
