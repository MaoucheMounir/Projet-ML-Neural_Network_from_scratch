import numpy as np
from Loss import *

class MSELoss(Loss):
    
    def __init__(self):
        return
    
    def forward(self, y, yhat):
        assert y.shape == yhat.shape
        
        return np.mean((y-yhat)**2, axis=0)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape
        return 2*(yhat-y)
    
    