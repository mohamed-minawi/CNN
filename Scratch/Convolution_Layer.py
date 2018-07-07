import numpy as np
from im2col import *
from im2col_cython import col2im_cython, im2col_cython
from im2col_cython import col2im_6d_cython
class Convolution_Layer:
    
    def __init__(self, input_dim, num_filters, filter_size, stride, weight_scale):
        self.depth, self.width, self.height = input_dim
        self.filter_size = filter_size
        self.stride = stride
        self.padding =  (self.filter_size - 1) // 2 
        self.num_filters = num_filters
        
        self.out_height = (self.height + 2 * self.padding - self.filter_size) / self.stride + 1
        self.out_width = (self.width + 2 * self.padding - self.filter_size) / self.stride + 1

        self.lr = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.fudge_factor = 1e-8
        self.timestamp = 0
        
        self.wght = weight_scale * np.random.randn(num_filters, self.depth, self.filter_size, self.filter_size)
        self.mo_w = np.zeros(self.wght.shape)
        self.acc_w = np.zeros(self.wght.shape)
        self.bias =  np.zeros(self.num_filters)
        self.mo_b = np.zeros(self.num_filters)
        self.acc_b = np.zeros(self.num_filters)

    def forward(self, x):
        N, C, H, W = x.shape
        F, _, HH, WW = self.wght.shape
        stride, pad = self.stride, self.padding
        w = self.wght
        b = self.bias

        # Check dimensions
        assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
        assert (H + 2 * pad - HH) % stride == 0, 'height does not work'

        # Pad the input
        p = pad
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        # Figure out output dimensions
        H += 2 * pad
        W += 2 * pad
        out_h = (H - HH) / stride + 1
        out_w = (W - WW) / stride + 1

        # Perform an im2col operation by picking clever strides
        shape = (C, HH, WW, N, out_h, out_w)
        strides = (H * W, W, 1, C * H * W, stride * W, stride)
        strides = x.itemsize * np.array(strides)
        x_stride = np.lib.stride_tricks.as_strided(x_padded,shape=shape, strides=strides)
        x_cols = np.ascontiguousarray(x_stride)
        x_cols.shape = (C * HH * WW, N * out_h * out_w)

        # Now all our convolutions are a big matrix multiply
        res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)

        # Reshape the output
        res.shape = (F, N, out_h, out_w)
        out = res.transpose(1, 0, 2, 3)

        # Be nice and return a contiguous array
        # The old version of conv_forward_fast doesn't do this, so for a fair
        # comparison we won't either
        out = np.ascontiguousarray(out)

        self. cache = (x, w, b, stride, pad, x_cols)
        return out

    def backward(self, dout):
        x, w, b, stride, pad, x_cols = self.cache

        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        _, _, out_h, out_w = dout.shape

        db = np.sum(dout, axis=(0, 2, 3))

        dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
        dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

        dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
        dx_cols.shape = (C, HH, WW, N, out_h, out_w)
        dx = col2im_6d_cython(dx_cols, N, C, H, W, HH, WW, pad, stride)

        self.dw = dw
        self.db = db

        return dx;

    def updateParameters(self):
    
        self.timestamp += 1
     
        self.mo_w = self.beta1 * self.mo_w + (1 - self.beta1) * self.dw
        self.mo_b = self.beta1 * self.mo_b + (1 - self.beta1) * self.db
        self.acc_w = self.beta2 * self.acc_w + (1 - self.beta2) * np.square(self.dw)
        self.acc_b = self.beta2 * self.acc_b + (1 - self.beta2) * np.square(self.db)
        
        mb = self.mo_b / (1 - self.beta1**self.timestamp)
        accb = self.acc_b / (1 - self.beta2**self.timestamp)
           
        mw = self.mo_w / (1 - self.beta1**self.timestamp)
        accw = self.acc_w / (1 - self.beta2**self.timestamp)

        
        self.wght -= self.lr * mw / (np.sqrt(accw) + self.fudge_factor) 
        self.bias -= self.lr * mb / (np.sqrt(accb) + self.fudge_factor)
    
    def regularization(self):
        return np.sum(self.wght**2)
 