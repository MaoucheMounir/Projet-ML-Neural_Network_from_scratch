from Abstract.Module import Module
import numpy as np
    

class ReLu(Module):
    def __init__(self,threshold=0.):
        self._threshold=threshold

    def forward(self, X):
        self._forward=self.threshold(X)
        return self._forward

    def zero_grad(self):
        pass
    
    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input:np.ndarray, delta:np.ndarray):
        pass

    def threshold(self,input):
        return np.where(input>self._threshold,input,0.)

    def derivative_Threshold(self,input):
        #Batch x out
        return (input > self._threshold).astype(float)

    def backward_delta(self, input, delta):
        self._delta=np.multiply(delta,self.derivative_Threshold(input))
        return self._delta