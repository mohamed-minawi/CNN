import numpy as np

class FC_Layer(object):
    
    def __init__(self, input_dim, output_dim, weight_scale, batchnorm, dropout):
        self.inpdim = input_dim
        self.outdim = output_dim
        self.batchnorm_flg = batchnorm
        self.dropout_flg = dropout > 0
        
        self.lr = 0.001
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.fudge_factor = 1e-8
        self.timestamp = 0
        
        self.wght = weight_scale * np.random.randn(self.inpdim, self.outdim)
        self.mo_w = np.zeros([self.inpdim,self.outdim])
        self.acc_w = np.zeros([self.inpdim,self.outdim])
        self.bias =  np.zeros(self.outdim)
        self.mo_b = np.zeros(self.outdim)
        self.acc_b = np.zeros(self.outdim)
       
    def forward(self, inp):
        activation = inp.reshape(inp.shape[0], self.wght.shape[0]).dot(self.wght)+self.bias
        self.mem = (inp, self.wght, self.bias)

        return activation

    def backward(self,dout):
        
        x, w, b = self.mem;
        dx, dw, db = None, None, None
        N = x.shape[0]
        dx = dout.dot(w.T).reshape(x.shape)
        dw = x.reshape(x.shape[0], self.wght.shape[0]).T.dot(dout)
        db = np.sum(dout,axis=0)
        self.dw = dw
        self.db = db

        return dx
    
    def updateParameters(self):

        self.timestamp += 1

        self.mo_w = self.beta1 * self.mo_w + (1 - self.beta1) * self.dw
        self.mo_b = self.beta1 * self.mo_b + (1 - self.beta1) * self.db
        self.acc_w = self.beta2 * self.acc_w + (1 - self.beta2) * np.square(self.dw)
        self.acc_b = self.beta2 * self.acc_b + (1 - self.beta2) * np.square(self.db)

        self.wght -= np.divide(self.lr * self.mo_w, (np.sqrt(self.acc_w) + self.fudge_factor))
        self.bias -= np.divide(self.lr * self.mo_b, (np.sqrt(self.acc_b) + self.fudge_factor))
        
    def regularization(self):
        return np.sum(self.wght**2)