import numpy as np

class Flatten:
    def __init__(self):
        pass
    def forward(self,X):
        output_sum = np.product(X.shape[1:])
        self.cache = X.shape
        return np.reshape(X,(X.shape[0],output_sum))
         
    def backward(self,dout):
        return np.reshape(dout,(self.cache))
