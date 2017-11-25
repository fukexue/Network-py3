from perceptron import Perceptron


def f(x):
    return 1 if x > 0 else 0


def get_training_dataset():
    input_vecs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    labels = [0, 0, 0, 1]
    return input_vecs, labels


def train_and_perceptron():
    P = Perceptron(2, f)
    input_vecs, labels = get_training_dataset()
    P.train(input_vecs, labels, 10, 0.1)
    return P


if __name__ == '__main__':
    and_perceptron = train_and_perceptron()
    print(and_perceptron)
    print('1 and 1 = %d' % and_perceptron.predict([1, 1]))
