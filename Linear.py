import numpy as np
from Module import *

class Linear(Module):
    
    def __init__(self, input:int, output:int):
        """_summary_

        Args:
            input (int): nombre d'entrées
            output (int): nombre de sorties
        
        Penser à eut etre ajouter le biais
        """
        
        self._parameters = np.random.rand(input, output)*100
        self._gradient = np.zeros((input, output))  #Car le gradient est de la forme nb_entrée*nb_sorties, et la loss a 1 seule sortie
        
    def forward(self, X):
        """Calcule la sortie à partir de X

        Args:
            X (_type_): _description_
        """
        input_dim = X.shape[1]
        assert self._parameters.shape[0] == input_dim
        
        return np.dot(X, self._parameters)
    
    def zero_grad(self):
        self._gradient = np.zeros(self._gradient.shape)
    
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## 
        """Calcule le gradient de la loss p.r aux paramètres
            et Met a jour la valeur du gradient

        Args:
            input (_type_): Les entrée du module
            delta ( np.array((nb_sorties_couche_courante,)) ): 
        """
        # la dérivée des sorties du module p.r aux parametres est l'input du module
        # La somme sur les k sorties se fait dans le produit matriciel
        # input=X car la dérivée se fait par rapport aux paramètres W
        self._gradient += np.dot(input.T, delta)  
    
    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur p.r aux entrées (sorties du module précédent)
        # input=Z_h-1 car la dérivée se fait par rapport aux Z de la couche precédente
        # Ca va être le delta qu'on va transmettre à la couche précédente
        # delta de la forme output*dim_loss (1 pour l'instant)
        return np.dot(delta, self._parameters.T)
    
    
    
    
    