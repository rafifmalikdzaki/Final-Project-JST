import numpy as np


class LVQ3:

    def __init__(self, alpha=0.1, epoch=100, beta=0.25, decay=0.1, m=0.15, dMethod="euclidean", epsilon=0.2, random_state=None):
        self.l_input = None
        self.o_input = None
        self.weight = None
        self.dMethod = dMethod
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.m = m
        self.epoch = epoch
        self.epsilon = epsilon
        if random_state:
            np.random.seed(random_state)

    def distanceM(self, input: np.ndarray) -> np.ndarray:
        if self.dMethod == "euclidean":
            return np.apply_along_axis(np.linalg.norm, 1, input-self.weight)

    def competitiveLayer(self, input: np.ndarray) -> [int, int, np.ndarray]:
        distances = self.distanceM(input)
        distanceArgs = np.argsort(distances)
        winner, runnerUp = distanceArgs[0], distanceArgs[1]
        return winner, runnerUp, distances[[winner, runnerUp]]

    def window(self, distances: np.ndarray) -> bool:
        if np.isin(distances, 0).any():
            return False
        ratios = (distances[0]/distances[1], distances[1]/distances[0])
        minRat = np.min(ratios)
        return minRat > ((1-self.epsilon)*(1+self.epsilon))

    def fit(self, train, target):
        X = np.array(train)
        y = np.array(target)

        self.l_input = X.shape[1]
        self.o_input = np.unique(y).shape[0]
        self.weight = np.random.uniform(low=-1, high=1, size=(self.o_input, self.l_input))

        epoch = 1
        while epoch <= self.epoch:
            for i, x in enumerate(X):
                winner, runnerUp, distances = self.competitiveLayer(x)
                target = y[i]
                left, right = (winner != target), (runnerUp != target)

                if left & right:
                    continue

                if self.window(distances):
                    if left ^ right:
                        if left:
                            yc1 = self.weight[runnerUp]
                            yc2 = self.weight[winner]

                            # Update weight
                            self.weight[runnerUp] = yc1 + self.alpha*(x - yc1)
                            self.weight[winner] = yc2 - self.alpha*(x - yc2)
                        elif right:
                            yc1 = self.weight[winner]
                            yc2 = self.weight[runnerUp]

                            # Update weight
                            self.weight[winner] = yc1 + self.alpha*(x - yc1)
                            self.weight[runnerUp] = yc2 - self.alpha*(x - yc2)

                    elif winner == target == runnerUp:
                        yc1 = self.weight[winner]
                        yc2 = self.weight[runnerUp]

                        # Update weight
                        self.weight[winner] = yc1 + self.beta*(x - yc1)
                        self.weight[runnerUp] = yc2 + self.beta*(x - yc2)

                        self.beta *= self.m * self.alpha

            self.alpha *= self.decay**epoch
            epoch += 1

    def predict(self, test) -> np.ndarray:
        test = np.array(test)
        result = []
        for x in test:
            winner, _, _ = self.competitiveLayer(x)
            result.append(winner)

        return np.array(result)
