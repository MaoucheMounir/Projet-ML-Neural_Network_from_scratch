from Abstract.Loss import *
import numpy as np

class CELoss(Loss):
    
    def __init__(self, name='CEs'):
        super().__init__()
        self._name = name
        
    def forward(self, y:np.ndarray, yhat:np.ndarray) -> float:
        
        # y_hat_truey + sum(exp(y_hat_i)) sur toutes les classes
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        assert y.shape == yhat.shape

        yhat_truey = np.sum(yhat*y, axis=1)
        
        return -yhat_truey 

    def zero_grad(self):
        pass
    
    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input:np.ndarray, delta:np.ndarray):
        pass
    
    def backward(self, y:np.ndarray, yhat:np.ndarray) -> np.ndarray:
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        assert y.shape == yhat.shape
        
        #return yhat*yhat_exp / np.sum(yhat_exp, axis=1).reshape((-1,1)) - y
        
        return yhat-y
        
        
        
        
    