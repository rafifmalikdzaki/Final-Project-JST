import numpy as np


class LVQ3:

    def __init__(self, alpha=0.1, beta=1, decay=0.1):
        self.l_input = None
        self.o_input = None
        self.weight = None
        self.alpha = alpha
        self.beta = beta
        self.decay = decay

    def distanceM(self, input: np.ndarray[np.double], method="euclidean") -> np.ndarray[np.double]:
        if method == "euclidean":
            return np.apply_along_axis(np.linalg.norm, 1, input-self.weight)
        pass

    def competitiveLayer(self, input: np.ndarray[np.double]) -> tuple[int]:
        pass

    def conditionCheck(self):
        pass

    def fit(self, train, target):
        pass
