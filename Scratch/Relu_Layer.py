import numpy as np

class Relu_Layer():
    
    def __init__(self):
        pass 
       
    def forward(self,inp):
        out = np.ones_like(inp)
        out[inp < 0] = 0.01 
        self.input_cache = inp
        return out * inp

    def backward(self,dout):
        X = self.input_cache
        dx = np.ones_like(X)
        dx[X < 0] = 0.01
        return dx * dout
    
