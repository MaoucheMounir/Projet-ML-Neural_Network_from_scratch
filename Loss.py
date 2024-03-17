import numpy as np
from abc import ABC
class Loss(ABC):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass