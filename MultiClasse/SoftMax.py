from Abstract.Module import Module
import numpy as np

class SoftMax(Module):
    
    def __init__(self, name='Softmax'):
        super().__init__()
        self._name = name
        
    def forward(self, input:np.ndarray) -> float:
        
        
        
        #exp = np.exp(np.sum(yhat*y, axis=1))
        y_hat_exp = np.exp(input)
        sums = np.sum(y_hat_exp, axis=1)
        nb_colonnes = input.shape[1]
        sums_repetitions = np.repeat(sums[:, np.newaxis], nb_colonnes, axis=1)

        
        return  y_hat_exp / sums_repetitions
        #return y_hat_exp / np.sum(y_hat_exp, axis=1)

    def zero_grad(self):
        pass
    
    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input:np.ndarray, delta:np.ndarray):
        pass
    
    def backward_delta(self, input:np.ndarray, delta:np.ndarray):
        softmax = self.forward(input)
        
        return delta * (softmax * (1 - softmax))
        
        
        
        
        
    