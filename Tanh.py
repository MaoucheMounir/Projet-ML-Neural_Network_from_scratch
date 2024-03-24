from Module import Module
import numpy as np

class Tanh(Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return np.tanh(x) # (1 - np.exp(-2*x)) / (1 + np.exp(-2*x))
    
    def zero_grad(self):
        self._gradient = np.zeros(self._gradient.shape)
    
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass

    def backward_update_gradient(self, input, delta):
        """Calcule le gradient de la loss p.r aux paramètres
            et Met a jour la valeur du gradient
        """
        pass
    
    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur p.r aux entrées (sorties du module précédent)
        # input=Z_h-1 car la dérivée se fait par rapport aux Z de la couche precédente
        # Ca va être le delta qu'on va transmettre à la couche précédente
        # delta de la forme output*dim_loss (1 pour l'instant)
        
        return delta * (1-self.forward(input)**2)
    
    