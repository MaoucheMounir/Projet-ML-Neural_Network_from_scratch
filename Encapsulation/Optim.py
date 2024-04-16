import numpy as np
from Abstract.Module import Module
from Abstract.Loss import Loss
from icecream import ic
from Encapsulation.Sequentiel import Sequentiel
from Lineaire.Linear import Linear

class Optim():
    
    def __init__(self, net:Sequentiel|Linear, loss:Loss, eps:float):
        """Optimiseur qui effectue une étape d'optimisation des paramètres pour le 
        réseau de neurones (calcule la loss et son gradient et mes à jour les paramètres)

        Args:
            net (Module): le réseau de neurone (module ou sequentiel)
            loss (function): la fonction cout
            eps (float): Pas de gradient
            
        Attributs supplémentaires:
            _cout = la liste des valeurs du cout
        """
        
        self._net:Sequentiel|Linear = net
        self._loss:Loss = loss 
        self._eps:float = eps
        self._couts:list[float] = []
    
    def step(self, batch_x:np.ndarray, batch_y:np.ndarray):
        output:np.ndarray = self._net.forward(batch_x) #fait
        cout:float = self._loss.forward(batch_y, output) #fait
        gradient_loss:np.ndarray = self._loss.backward(batch_y, output) # fait, de taille (100,1)
        
        # self._net.backward_delta(batch_x, gradient_loss)
        # self._net.backward_update_gradient(batch_x, gradient_loss)
        self._net.backward(batch_x, gradient_loss)
        self._net.update_parameters(self._eps)
        self._couts.append(cout)
        
    
    def score(self, X, Y):
        return np.where(Y == self._net.forward(X), 1, 0).mean()