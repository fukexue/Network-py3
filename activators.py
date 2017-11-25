import math
import numpy as np


class SigmoidActivator(object):
    def forward1(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)
