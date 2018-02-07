import numpy as np
import pandas as pd

from Adaline.plot_utils import plot_error_function, plot_decision_regions


class AdaptiveLinearNeuron:
    def learn(self, data, learning_rate=0.01, epochs=10):
        x, y = data
        self.weights = np.zeros(1 + x.shape[1])

        f_error = []
        for i in range(epochs):
            output = self._calc_net_input(x)
            errors = y - output
            self.weights[1:] += learning_rate * x.T.dot(errors)
            self.weights[0] += learning_rate * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            f_error.append(cost)
        return f_error

    def _calc_net_input(self, x):
        return np.dot(x, self.weights[1:]) + self.weights[0]

    def predict(self, x):
        return np.where(self._calc_net_input(x) >= 0.0, 1, -1)


def standardize(x):
    """
    feature scaling: standartization
    x - mean / standard deviation
    this helps with gradient descent learning, to go through fewer epochs
    """
    x_std = np.copy(x)
    x_std[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
    x_std[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()
    return x_std


learning_data = pd.read_csv("learning_data.csv")

Y = learning_data.iloc[0:150, 4].values
Y = np.where(Y == 'Iris-setosa', -1, 1)  # replace Iris-setosa with 1 else -1
X = learning_data.iloc[0:150, [0, 2]].values  # take columns at index 0 and 2
X = standardize(X)

aln = AdaptiveLinearNeuron()
f_error = aln.learn(data=(X, Y), learning_rate=0.001, epochs=100)

plot_error_function(f_error)
plot_decision_regions(X, Y, classifier=aln)
