from functools import reduce
import math
import random
from datetime import datetime


def sigmoid(inx):
    return 1.0 / (1.0 + math.exp(-inx))


class Node(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        self.output = output

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        self.upstream.append(conn)

    def calc_output(self):
        output = reduce(lambda ret, conn: ret+conn.upstream_node.output*conn.weight, self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(lambda ret, conn: ret+conn.downstream_node.output*conn.weight, self.downstream, 0)
        self.delta = self.output*(1-self.output)*downstream_delta

    def calc_output_layer_delta(self, label):
        self.delta = self.output*(1-self.output)*(label - self.output)

    def __str__(self):
        node_str = '%u-%u output:%f,delta:%f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


class ConstNode(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_dowmstream_connection(self, conn):
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(lambda ret, conn: ret+conn.downstream_node*conn.weight, self.downstream, 0)
        self.delta = self.output*(1-self.output)*downstream_delta

    def __str__(self):
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


class Layer(object):
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        for node in self.nodes:
            print(node)


class Connection(object):
    def __init__(self, downstream_node, upstream_node):
        self.downstream_node = downstream_node
        self.upstream_node = upstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0

    def calc_gradient(self):
        return self.gradient

    def update_weight(self, rate):
        self.calc_gradient()
        self.weight += rate * self.gradient

    def __str__(self):
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)


class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print(conn)


class Network(object):
    def __init__(self, layers):
            self.connections = Connections()
            self.layers = []
            layer_count = len(layers)
            node_count = 0
            for i in range(layer_count):
                self.layers.append(Layer(i, layers[i]))
            for layer in range(layer_count-1):
                connections_zip = zip(self.layers[layer].nodes, self.layers[layer+1].nodes[:-1])
                connections = map(lambda stream_node: Connection(stream_node[0], stream_node[1]), connections_zip)
                # 在python2.7中采用for for的形式
                for conn in connections:
                    self.connections.add_connection(conn)
                    conn.downstream_node.append_upstream_connection(conn)
                    conn.upstream_node.append_downstream_connection(conn)

    def train(self, label, date_set, rate, iteration):
        for i in range(iteration):
            for d in range(len(date_set)):
                self.train_one_sample(label[d], date_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return list(map(lambda node: node.output, self.layers[-1].nodes[:-1]))

    def calc_delta(self, label):
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    # def loss(self, output, label):
    #     return 0.5*((label - output)*(label - output)).sum()

    def dump(self):
        for layer in self.layers:
            layer.dump()


def mean_square_error(vec1, vec2):
    return 0.5 * reduce(lambda a, b: a + b,
                        map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                            zip(vec1, vec2)))


def gradient_check(network, sample_feature, sample_label):
    network.get_gradient(sample_feature, sample_label)
    for conn in network.connections.connections:
        actual_gradient = conn.get_gradient()
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = mean_square_error(network.perdict(sample_feature), sample_label)
        conn.weight -= 2*epsilon
        error2 = mean_square_error(network.perdict(sample_feature), sample_label)

        expected_gradient = (error1 - error2)/(2*epsilon)

        print('expected gradient: \t%f\nactual gradient: \t%f' % (expected_gradient, actual_gradient))


class Normalizer(object):
    def __init__(self):
        self.mask = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]

    def norm(self, number):
        return map(lambda m: 0.9 if number & m else 0.1, self.mask)

    def denorm(self, vec):
        binary = map(lambda i: 1 if i > 0.5 else 0, vec)
        return reduce(lambda x, y: x + y, map(lambda a, b: a*b, binary, self.mask), 0)


def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))
