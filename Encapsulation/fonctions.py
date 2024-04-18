import numpy as np
from Encapsulation.Optim import Optim
from Abstract.Loss import Loss
from icecream import ic

def SGD(net, X:np.ndarray, Y:np.ndarray, nb_batch:int, loss:Loss, nb_epochs=10, eps:float=1e-5, shuffle:bool=False):
    """Effectue la descente de gradient stochastique/batch.

    Args:
        net (Module): Le réseau de neurone ou le module
        X (np.ndarray): L'ensemble des exemples de train
        Y (np.ndarray): L'ensemble des labels de train
        taille_batch (int): La taille de chaque batch
        loss (Function): La fonction de cout
        nb_iter (int, optional): Nombre d'itérations. Defaults to 100.
        eps (float, optional): Pas de gradient. Defaults to 1e-3.
        shuffle (bool, optional): Si permuter les exemples ou non. Defaults to False.

    Returns:
        optim._couts : La liste des couts calculés par l'optimiseur
        net : Le réseau de neurones entraîné
    """
    
    #Y = np.reshape(Y, (-1, 1))

    X_Y = np.hstack((X, Y))
    
    if shuffle:
        np.random.shuffle(X_Y)
    
    optim = Optim(net, loss, eps)
    batches = np.array_split(np.array(X_Y), nb_batch)
    for epoch in range(nb_epochs):
        
        for batch in batches:
            
            batch_x = np.array([b[:-Y.shape[1]] for b in batch]) #Modifié ca, c'etait -1, pour généraliser
            batch_y = np.array([b[-Y.shape[1]:] for b in batch])
            
            optim.step(batch_x, batch_y)
    
    return optim._net, optim._couts, optim
    