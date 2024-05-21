from Abstract import Module
import numpy as np

class Conv1D():

    def __init__(self, k_size, chan_in, chan_out, stride=1, bias=True):
        """
        k_size : kernel size
        chan_in : Nombre de canaux (in)
        chan_out : Nombre de canaux (out)

        """
        #super(Conv1D, self).__init__()
        self.k_size=k_size
        self.chan_in=chan_in
        self.chan_out=chan_out
        self.stride=stride
        b=1 / np.sqrt(chan_in*k_size)
        self._parameters = np.random.uniform(-b, b, (k_size,chan_in,chan_out))
        self._gradient=np.zeros(self._parameters.shape)
        self.bias = bias
        if(self.bias):
            self._bias=np.random.uniform(-b, b, chan_out)
            self._gradBias = np.zeros((chan_out))

    def zero_grad(self):
        self._gradient=np.zeros(self._gradient.shape)
        if (self.bias):
            self._gradBias = np.zeros(self._gradBias.shape)

    def forward(self, X):
        """
        X: (batch,input,chan_in)
        out: (batch, (input-k_size)/stride +1,chan_out)
        """
        size = ((X.shape[1] - self.k_size) // self.stride) + 1

        out=np.array([(X[:, i: i + self.k_size, :].reshape(X.shape[0], -1)) @ (self._parameters.reshape(-1, self.chan_out))
                        for i in range(0,size,self.stride)])
        if (self.bias):
            out+=self._bias
        self._forward=out.transpose(1,0,2)
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step * self._gradient
        if self.bias:
            self._bias -= gradient_step * self._gradBias

    def backward_update_gradient(self, input, delta):
        """
        input: (batch,input,chan_in)
        delta: (batch, (input-k_size)/stride +1,chan_out)
        """
        size = ((input.shape[1] - self.k_size) // self.stride) + 1
        out = np.array([ (delta[:,i,:].T) @ (input[:, i: i + self.k_size, :].reshape(input.shape[0], -1))  \
                        for i in range(0, size, self.stride)])
        self._gradient=np.sum(out,axis=0).T.reshape(self._gradient.shape)/delta.shape[0]

        if self.bias:
            self._gradBias=delta.mean((0,1))

    def backward_delta(self, input, delta):
        """
        input: (batch,input,chan_in)
        delta: (batch, (input-k_size)/stride +1,chan_out)
        out: (batch,input,chan_in)
        """
        size = ((input.shape[1] - self.k_size) // self.stride) + 1
        out = np.zeros(input.shape)
        for i in range(0, size, self.stride):
            out[:,i:i+self.k_size,:] += ((delta[:, i, :]) @ (self._parameters.reshape(-1,self.chan_out).T)).reshape(input.shape[0],self.k_size,self.chan_in)
        self._delta= out
        return self._delta


class MaxPool1D(): #Module

    def __init__(self, k_size=3, stride=1):
        #super(MaxPool1D, self).__init__()
        self.k_size = k_size
        self.stride = stride
        self._parameters = None
        
    def forward(self, X):
        """
        X: (batch,input,chan_in)
        out:  (batch,(input-k_size)/stride +1,chan_in)
        """

        size = ((X.shape[1] - self.k_size) // self.stride) + 1
        out = np.zeros((X.shape[0], size, X.shape[2]))
        for i in range(0, size, self.stride):
            out[:,i,:]=np.max(X[:,i:i+self.k_size,:],axis=1)
        self._forward=out
        return self._forward
    
    def zero_grad(self):
        pass

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_delta(self, input, delta):
        """
         input: (batch,input,chan_in)
         delta: (batch,(input-k_size)/stride +1,chan_in)
         out: (batch,input,chan_in)
        """
        size = ((input.shape[1] - self.k_size) // self.stride) + 1
        out=np.zeros(input.shape)
        batch=input.shape[0]
        chan_in=input.shape[2]
        for i in range(0,size,self.stride):
            indexes_argmax = np.argmax(input[:, i:i+self.k_size,:], axis=1) + i
            out[np.repeat(range(batch),chan_in),indexes_argmax.flatten(),list(range(chan_in))*batch]=delta[:,i,:].reshape(-1)
        self._delta=out
        return self._delta
    
    def backward_update_gradient(self, input, delta):
        pass
    
    
    
    
class Flatten():

    def __init__(self):
        #super(Flatten, self).__init__()
        self._parameters = None
        pass
    
    def zero_grad(self):
        pass

    def forward(self, X):
        """
        X:(batch,input,chan_in)
        out:(batch,input*chan_in)
        """
        self._forward = X.reshape(X.shape[0], -1)
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_delta(self, input, delta):
        """
         input: (batch,input,chan_in)
         delta: (batch, input * chan_in)
         out: (batch,input,chan_in)
        """
        self._delta = delta.reshape(input.shape)
        return self._delta
    
    def backward_update_gradient(self, input, delta):
        pass
    
    
####################################################################################################################################################################################


class Conv2D():
    def __init__(self, k_size, chan_in, chan_out, stride=1, bias=True):
        self.k_size=k_size
        self.chan_in=chan_in
        self.chan_out=chan_out
        self.stride=stride
        b=1 / np.sqrt(chan_in*k_size)
        
        self._parameters = np.random.uniform(-b, b, (k_size,k_size,chan_in,chan_out))
        self._gradient=np.zeros(self._parameters.shape)
        self.bias = bias
        if(self.bias):
            self._bias=np.random.uniform(-b, b, chan_out)
            self._gradBias = np.zeros((chan_out))

    def zero_grad(self):
        self._gradient = np.zeros(self._gradient.shape)
        if (self.bias):
            self._gradBias = np.zeros(self._gradBias.shape)

    def forward(self, X):
        """
         X:(batch, H, W, chan_in)
         out:(batch,size_H,size_W,chan_out)
        """

        size_h = ((X.shape[1] - self.k_size) // self.stride) + 1
        size_w = ((X.shape[2] - self.k_size) // self.stride) + 1
        out=np.zeros((X.shape[0],size_h,size_w,self.chan_out))

        for i in range(0,size_h,self.stride):
            for j in range(0,size_w,self.stride):
                out[:,i,j,:]=X[:,i: i + self.k_size,j: j + self.k_size,:].reshape((X.shape[0],-1)) @ self._parameters.reshape(-1,self.chan_out)
        if (self.bias):
            out += self._bias
        self._forward = out
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
       
        self._parameters -= gradient_step * self._gradient
        if self.bias:
            self._bias -= gradient_step * self._gradBias

    def backward_update_gradient(self, input, delta):
      
        size_h = ((input.shape[1] - self.k_size) // self.stride) + 1
        size_w = ((input.shape[2] - self.k_size) // self.stride) + 1
        out=np.zeros(self._gradient.shape)
        for i in range(0, size_h, self.stride):
            for j in range(0, size_w, self.stride):
                    s=(delta[:, i,j, :].T) @ (input[:, i: i + self.k_size,j: j + self.k_size, :].reshape(input.shape[0], -1))
                    out+=s.reshape(s.shape[0],self.k_size,self.k_size,input.shape[-1]).transpose(1,2,3,0)
        self._gradient=out/delta.shape[0]
        if self.bias:
            self._gradBias=delta.mean((0,1,2))

    def backward_delta(self, input, delta):
        size_h = ((input.shape[1] - self.k_size) // self.stride) + 1
        size_w = ((input.shape[2] - self.k_size) // self.stride) + 1
        out = np.zeros(input.shape)
        for i in range(0, size_h, self.stride):
            for j in range(0, size_w, self.stride):
                s=(delta[:,i,j,:])@(self._parameters.reshape(-1,self.chan_out).T)
               
                out[:, i:i + self.k_size, j:j + self.k_size, :]+=s.reshape(delta.shape[0],self.k_size,self.k_size,self.chan_in)
                
        self._delta=out
        return self._delta


class MaxPool2D():

    def __init__(self, k_size=3, stride=1):
        self.k_size = k_size
        self.stride = stride

    def forward(self, X):
        """
        X: (batch,Hlength,Wlength,chan_in)
        out :  (batch,t_h,t_w,chan_in)
        """
        t_h = ((X.shape[1] - self.k_size) // self.stride) + 1
        t_w = ((X.shape[2] - self.k_size) // self.stride) + 1
        sortie=np.zeros((X.shape[0],t_h,t_w,X.shape[-1]))
        for i in range(0, t_h, self.stride):
            for j in range(0, t_w, self.stride):
                sortie[:, i,j, :] = np.max(X[:,i:i+self.k_size, j:j + self.k_size, :], axis=(1,2))
        self._forward=sortie
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_delta(self, input, delta):
        t_h = ((input.shape[1] - self.k_size) // self.stride) + 1
        t_w = ((input.shape[2] - self.k_size) // self.stride) + 1
        batch=input.shape[0]
        chan_in=input.shape[3]
        sortie=np.zeros((input.shape[0],input.shape[1]*input.shape[2],input.shape[3]))
        for i in range(0, t_h, self.stride):
            for j in range(0, t_w, self.stride):
                region_shaped=input[:, i:i+self.k_size,j:j+self.k_size,:].reshape(-1,self.k_size*self.k_size,chan_in)
                idx_argmax=np.argmax(region_shaped, axis=1) + i + j
                sortie[np.repeat(range(batch),chan_in),idx_argmax.flatten(),list(range(chan_in))*batch]=delta[:,i,j,:].reshape(-1)
        self._delta=sortie.reshape(input.shape)
        return self._delta
    
    def zero_grad():
        pass
    
    def backward_update_gradient(self, input, delta):
        pass
