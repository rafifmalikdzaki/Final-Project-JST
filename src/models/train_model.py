import numpy as np


class LVQ3:

    def __init__(self, alpha=0.1, beta=1, decay=0.1, dMethod="euclidean", epsilon=0.2):
        self.l_input = None
        self.o_input = None
        self.weight = None
        self.dMethod = dMethod
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon

    def distanceM(self, input: np.array[np.double]) -> np.array[np.double]:
        if self.dMethod == "euclidean":
            return np.apply_along_axis(np.linalg.norm, 1, input-self.weight)
        pass

    def competitiveLayer(self, input: np.array[np.double]) -> [int, int]:
        distances = self.distanceM(input)
        distanceArgs = np.argsort(distances)
        winner, runnerUp = distanceArgs[0], distanceArgs[1]
        return winner, runnerUp, distances[:2]

    def window(self, distances: np.array[np.double]) -> bool:
        ratios = (distances[0]/distances[1], distances[1]/distances[0])
        minRat = np.min(ratios)
        return minRat > ((1-self.epsilon)*(1+self.epsilon))

    def fit(self, train, target):
        X = np.array(train)
        y = np.array(target) + 1

        self.l_input = X.shape[0]
        self.o_input = y.shape[0]
        self.weight = np.random.uniform(
            low=-1, high=1, size=(self.o_input, self.l_input))

        for i, x in enumerate(X):
            winner, runnerUp, distances = self.competitiveLayer(x)
            target = y[i]
            left, right = (winner != target), (runnerUp != target)

            if left & right:
                continue

            if self.window(distances):
                if left ^ right:
                    continue
                elif winner == target == runnerUp:
                    continue

        pass
