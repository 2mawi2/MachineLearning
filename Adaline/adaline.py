import numpy as np
import pandas as pd

from Adaline.plot_utils import plot_error_function, plot_decision_regions


class AdaptiveLinearNeuron:
    def learn(self, data, a=0.01, epochs=10):
        """
        Q = sample data
        E = sum squared error
        e = error
        t = signal function (-1 / 1)
        a = alpha - learning rate
        """
        X, t = data

        self.w = np.zeros(1 + X.shape[1])  # init all weights to zero

        error_function = []
        for _ in range(epochs):
            e = t - self.y(X)
            self.w[1:] += a * X.T.dot(e)
            self.w[0] += a * e.sum()
            E = (1 / 2) * (e ** 2).sum()
            error_function.append(E)
        return error_function

    def y(self, x):
        return np.dot(x, self.w[1:]) + self.w[0]

    def predict(self, x):
        return np.where(self.y(x) >= 0.0, 1, -1)  # sigma signal function 1 or -1


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


learning_data = pd.read_csv("./learning_data.csv")

t = learning_data.iloc[0:150, 4].values
t = np.where(t == 'Iris-setosa', -1, 1)  # replace Iris-setosa with 1 else -1

Q = learning_data.iloc[0:150, [0, 1]].values  # take columns at index 0 and 2
# X = standardize(X)

aln = AdaptiveLinearNeuron()
f_error = aln.learn(data=(Q, t), a=0.0001, epochs=1000)

plot_error_function(f_error)
# plot_decision_regions(X, Q, classifier=aln)
