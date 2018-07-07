import numpy as np

class Softmax_Layer:
    def __init__(self):
        pass

    def forward(self, X):
        self.cache = X
        probs = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.Y = probs/np.sum(probs, axis=1, keepdims=True)
        return self.Y

    def backward(self, x, y):
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        N = x.shape[0]
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx