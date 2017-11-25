from perceptron import Perceptron


def f_liner(x):
    return x


class LinerUint(Perceptron):
    def __init__(self, input_num):
        Perceptron.__init__(self, input_num, f_liner)


def get_training_database():
    input_vecs = [[5, 3, 5, 7], [3, 6, 4, 2], [8, 6, 3, 6], [1.4, 12, 4, 6], [10.1, 6, 4, 3]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels


def train_liner_uint():
    liner = LinerUint(4)
    input_vecs, labels = get_training_database()
    liner.train(input_vecs, labels, 10, 0.01)
    return liner


if __name__ == '__main__':
    liner_uint = train_liner_uint()
    print(liner_uint)

    print('Work 3.4 years, monthly salary = %.2f' % liner_uint.predict([12, 3, 7, 7]))
