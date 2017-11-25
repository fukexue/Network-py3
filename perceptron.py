from functools import reduce
#
# 感知器模型


class Perceptron(object):
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        return 'weight:%s\tbias:%f' % (list(self.weights), self.bias)

    def predict(self, input_vec):
        temp = map(lambda x: x[0]*x[1], zip(input_vec, self.weights))
        return self.activator(reduce(lambda a, b: a+b, temp, 0.0) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, label, output, rate)

    def _update_weights(self, input_vec, label, output, rate):
        delta = label - output
        self.weights = list(map(lambda x: x[1] + rate*delta*x[0], zip(input_vec, self.weights)))
        print(self.weights)
        self.bias = self.bias + rate*delta
