import numpy as np
from Loss import *

class MSELoss(Loss):
    
    def __init__(self):
        return
    
    def forward(self, y, yhat):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        assert y.shape == yhat.shape

        return np.mean((y-yhat)**2, axis=0)

    def backward(self, y, yhat):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        assert y.shape == yhat.shape
        return 2*(yhat-y)
    
    