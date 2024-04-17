
class Sequentiel :
    
    def __init__(self,modules,label=None):
        self._moduls=modules
        self._features=[]
        #sert si on fait une regression lineare ou non
        self._label=label

    def forward(self, X):
        """
            avoir en premier les premiere features puis les new_features transformer grace a la premiere couche puis la 2eme ect jusqu'a la derniere couche
        """

        self._features=[X]
        for modul in self._moduls :
            X=modul.forward(X)
            self._features.append(X)

        return self._features[-1]
        
    def backward(self, delta ):
        """
            fait backward en entier : backward_delta + backward_update_gradient
        """

        features_rev=self._features[::-1]

        for i,modul in enumerate( self._moduls[::-1] ):
            input=features_rev[1+i]
            modul.backward_update_gradient(input,delta)
            delta=modul.backward_delta(input,delta)
        
    def update_parameters(self, gradient_step=1e-3):
        """
            fait la maj du discente du gradient
        """
        
        for modul in self._moduls :
            modul.update_parameters(gradient_step)
            modul.zero_grad()

    def predict(self,X):
        if self._label is None : 
            return self.forward(X)
        return self._label(self.forward(X))

