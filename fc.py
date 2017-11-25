from activators import SigmoidActivator
import numpy as np
import numpy.random as npr


class FullconnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        self.W = npr.uniform(-0.1, 0.1, (output_size, input_size))
        self.b = np.zeros((output_size, 1))
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        self.input = input_array
        tttt = np.dot(self.W, input_array)
        self.output = self.activator.forward1(self, tttt)

    def backward(self, delta_array):
        self.delta = self.activator.backward(self, output=self.input)*np.dot(self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        self.W += learning_rate*self.W_grad
        self.b += learning_rate*self.b_grad

    def dump(self):
        print('w:%s,\tb%s' % (self.W, self.b))


class Network(object):
    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(FullconnectedLayer(layers[i], layers[i+1], SigmoidActivator))

    def predict(self, sample):
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def calc_gradient(self, label):
        label_tr = np.array(label, ndmin=2).T
        rrr = self.layers[-1].output
        delta = self.layers[-1].activator.backward(self, output=rrr)*(label_tr - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def train(self, labels, samples, rate, epoch):
        for i in range(epoch):
            for d in range(len(samples)):
                sample_tr = np.array(samples[d], ndmin=2).T
                self.train_one_sample(labels[d], sample_tr, rate)

    def loss(self, output, label):
        return 0.5*((label - output)*(label - output)).sum()

    def dump(self):
        for layer in self.layers:
            layer.dump()

    def gradient_check(self, sample_feature, sample_label):
        self.predict(sample_feature)
        self.calc_gradient(sample_label)

        epsilon = 10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i, j] += epsilon
                    output = self.predict(sample_feature)
                    err1 = self.loss(sample_label, output)
                    fc.W[i, j] -= 2*epsilon
                    output = self.predict(sample_feature)
                    err2 = self.loss(sample_label, output)
                    expect_grad = (err1 - err2) / (2 * epsilon)
                    fc.W[i, j] += epsilon
                    print('weights(%d,%d): expected - actural %.4e - %.4e' % (i, j, expect_grad, fc.W_grad[i, j]))


def transpose(args):
    return map(
        lambda arg: map(
            lambda line: np.array(line).reshape(len(line), 1), arg), args)


