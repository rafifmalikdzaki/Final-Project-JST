import numpy as np


class LVQ3:

    def __init__(self, alpha=0.1, beta=1, decay=0.1):
        self.l_input = None
        self.o_input = None
        self.weight = None
        self.alpha = alpha
        self.beta = beta
        self.decay = decay

    def fit(self, train, target):
        pass
