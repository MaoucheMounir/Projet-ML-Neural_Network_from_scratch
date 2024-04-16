import numpy as np
from Abstract.Module import *
from icecream import ic
import pandas as pd

class Sequentiel(Module):
    
    def __init__(self, *args):
        """
        Arguments:
            *args (Module): Les couches du réseau dans l'ordre
            
        Attributs:
            _modules (list[Modules]): La liste des couches du réseau dans l'ordre positif
            _outputs (list[np.ndarray]): La liste des outputs des couches dans l'ordre positif
            _deltas (list[np.ndarray]): La liste des deltas dans le sens inverse des couches (sens négatif)
        """
        
        self._modules:list[Module] = list(args)
        
        self._outputs:list[np.ndarray] = []
        self.verifier_modules()
        self._deltas:list[np.ndarray] = []
    
    def verifier_modules(self): ##!! A completer
        try:
            pass #Verifier les dimensions
        except:
            raise ValueError('dimns incompatibles')
    
    def forward(self, X:np.ndarray):
        """Calcule la sortie à partir de X

        Args:
            X (np.ndarray): Les données d'entrée
        """
        
        output:np.ndarray = X
        self._outputs.append(output)
        
        for module in self._modules:  
            self._outputs.append(module.forward(self._outputs[-1]))
        return self._outputs[-1]
    
    def zero_grad(self):
        for module in self._modules:
            module.zero_grad()
    
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        #self._parameters -= gradient_step*self._gradient
        
        for module in self._modules:
            module.update_parameters(gradient_step)
            module.zero_grad()
    
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
        #self._gradient += np.dot(input.T, delta)  
        
        outputs_reversed = self._outputs[::-1]
        
        for i, module in enumerate(self._modules[::-1]):
            module.backward_update_gradient(outputs_reversed[i+1], self._deltas[i])
    
    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur p.r aux entrées (sorties du module précédent)
        # input=Z_h-1 car la dérivée se fait par rapport aux Z de la couche precédente
        # Ca va être le delta qu'on va transmettre à la couche précédente
        # delta de la forme output*dim_loss (1 pour l'instant)
        #return np.dot(delta, self._parameters.T)
        
        delta_local = delta
        self._deltas.append(delta_local)
        outputs_reversed = self._outputs[::-1]
        
        for i, module in enumerate(self._modules[::-1]):
            delta_local = module.backward_delta(outputs_reversed[i+1], self._deltas[-1])
            self._deltas.append(delta_local) # ii on a enlevé axis
            
        #La liste des deltas est dans le sens inverse de celle des modules
    
    
    def backward(self, input, delta):
    
        """Calcule le gradient de la loss p.r aux paramètres
            et Met a jour la valeur du gradient

        Args:
            input (_type_): Les entrée du module
            delta ( np.array((nb_sorties_couche_courante,)) ): 
        """
        # la dérivée des sorties du module p.r aux parametres est l'input du module
        # La somme sur les k sorties se fait dans le produit matriciel
        # input=X car la dérivée se fait par rapport aux paramètres W
        #self._gradient += np.dot(input.T, delta)  
        
        outputs_reversed = self._outputs[::-1]
        
        for i, module in enumerate(self._modules[::-1]):
            module.backward_update_gradient(outputs_reversed[i+1], delta)
            delta=module.backward_delta(outputs_reversed[i+1], delta)
            
        
    # def predict(self, X):
    #     return np.where(self.forward(X)>=0.5, 1, 0)
    
    def reset(self):
        for module in self._modules:
            module.reset()
    
    def describe_shape(self):
        noms = [module._name for module in self._modules]
        inputs_dim = [module._input_dim for module in self._modules]
        outputs_dim = [module._output_dim for module in self._modules]
        
        dims_parametres = [module._parameters.shape if module._parameters is not None else None for module in self._modules]
        dims_gradients = [module._gradient.shape if module._gradient is not None else None for module in self._modules]
        df = pd.DataFrame(list(zip(noms, inputs_dim, outputs_dim, dims_parametres, dims_gradients)), columns=['noms', 'inputs_dim', 'outputs_dim', 'dims_parametres', 'dims_gradients'])
        print(df)
        
    def describe_values(self):
        noms = [module._name for module in self._modules]
        parametres = [module._parameters[0] if module._parameters is not None else None for module in self._modules]
        gradients = [module._gradient[0] if module._gradient is not None else None for module in self._modules]
        df = pd.DataFrame(list(zip(noms,parametres, gradients)), columns=['noms', 'parametres', 'gradients'])
        print(df)
        
