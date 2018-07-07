import numpy as np
from FC_Layer import *
from Convolution_Layer import *
from Pooling_Layer import *
from Relu_Layer import *
from Softmax_Layer import *

class NN(object):
    def __init__(self, reg):
        self.reg = reg
        self.layers = []

    def loss(self, X_inp, Y_inp = None):
        
        dx = {}
        out = X_inp
        loss = 0.0

        for layer in self.layers:
            if not isinstance(layer, Softmax_Layer):
                out = layer.forward(out)
                
        scores = out
        if Y_inp is None:
            return scores
        
        loss, dscores = self.layers[-1].backward(scores, Y_inp)
        
        for layer in reversed(self.layers):
            if not isinstance(layer, Softmax_Layer):
                dscores = layer.backward(dscores)
                
        for layer in self.layers:
            if isinstance(layer, FC_Layer) or isinstance(layer, Convolution_Layer):
                loss += 0.5 * self.reg * layer.regularization()
                layer.dw += self.reg * layer.wght

        return loss
    
    def add_Layer(self,new_layer, lr= 0.01):
        new_layer.lr = lr
        self.layers.append(new_layer)

    def update_Layer(self):
        for layer in self.layers:
            if isinstance(layer, FC_Layer) or isinstance(layer, Convolution_Layer):
                layer.updateParameters()
