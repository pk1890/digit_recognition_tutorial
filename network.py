
import random
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d

class Network(object):
    def __init__(self, layers_sizes):
        self.num_layers = len(layers_sizes)
        self.layers_sizes = layers_sizes
        self.biases = [np.random.randn(y, 1) for y in layers_sizes[1:]]
        self.weights = [np.random.randn(y, x) for y, x in zip(layers_sizes[1:], layers_sizes[:-1])] # okreslam macierz wag dla kazdej warstwy tak, ze w[l][y][x] to waga polaczenia z neurona nr.x z warstwy l-1 do neurona nr.y z warstwy l
    
    def feedforward(self, data):
        out = data
        for weight, bias in zip(self.weights, self.biases):
            out = sigmoid(np.dot(weight, out) + bias)
        return out

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def learn_on_mnist(self, eta, epochs, batch_size):
        import mnist_loader
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        self.SGD(training_data, eta, epochs, batch_size, test_data=test_data)

    def run(self, imgPath):
        data = loadImg(imgPath)
        return np.argmax(self.feedforward(data))

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def SGD(self, trainingData, eta, epochsNumber, dataBatchSize, test_data=None):
        if test_data: n_test = len(test_data)
        n = len(trainingData)

        for j in xrange(epochsNumber):
            random.shuffle(trainingData)
            dataBatches = [trainingData[k:k+dataBatchSize] for k in xrange(0, n, dataBatchSize)]
            for dataBatch in dataBatches:
                self.updateDataBatch(dataBatch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)
            
    def updateDataBatch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                    for b, nb in zip(self.biases, nabla_b)]

def loadImg(name):
    img = Image.open("./" + name)
    p = np.array(img)
    p.reshape(-1, p.shape[2])
    m = interp1d([0, 255], [1, 0])
    p = np.float32(m([t[0] for t in p.reshape(-1, p.shape[2])])).reshape(-1,1)
    return p

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

   
