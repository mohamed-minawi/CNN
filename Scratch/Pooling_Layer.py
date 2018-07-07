import numpy as np
from im2col import *

class Pooling_Layer(object):
      
    def __init__(self, pool_height = 2 , pool_width = 2, stride = 2):
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride
        
    def forward(self, x):
        N, C, H, W = x.shape
        pool_height, pool_width = self.pool_height, self.pool_width
        stride = self.stride
        assert pool_height == pool_width == stride, 'Invalid pool params'
        assert H % pool_height == 0
        assert W % pool_height == 0
        x_reshaped = x.reshape(N, C, H / pool_height, pool_height,  W / pool_width, pool_width)
        out = x_reshaped.max(axis=3).max(axis=4)

        self.cache = (x, x_reshaped, out)
        return out

    def backward(self, dout):
        
        x, x_reshaped, out = self.cache
        dx_reshaped = np.zeros_like(x_reshaped)
        out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
        mask = (x_reshaped == out_newaxis)
        dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
        dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
        dx_reshaped[mask] = dout_broadcast[mask]
        dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
        dx = dx_reshaped.reshape(x.shape)

        return dx
      
