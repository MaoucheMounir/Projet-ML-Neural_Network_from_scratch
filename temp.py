import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from Lineaire.Linear import *
from Loss.BCELoss import BCELoss
from Activation.Tanh import Tanh
from Activation.Sigmoide import Sigmoide
from Activation.ReLu import ReLU
from Encapsulation.AutoEncodeur import AutoEncodeur
from Encapsulation.Sequentiel import Sequentiel
from Encapsulation.Optim import SGD
from Activation.SoftMax import  SoftMax
from Loss.CELogSoftMax import CELogSoftMax

from utils import tools

from icecream import ic


def transform_one_hot_vector(classe):
    """Retourne l'encodage one-hot d'une classe

    Args:
        classe (int): Le numéro de la classe

    Returns:
        Le vecteur one-hot de la classe
    """
    a =  np.zeros(10)
    a[classe] = 1
    return a

def transform_one_hot(Y):
    """Transforme l'ensemble des labels en vecteurs one-hot

    Args:
        Y (np.ndarray): L'ensemble des labels

    Returns:
        Matrice des labels one-hot
    """
    one_hot_vectors = []
    for yi in Y:
        one_hot_vectors.append(transform_one_hot_vector(yi))
    return np.array(one_hot_vectors)

def pred_classes(y_hat):
    """Retourne les classes à prédire à partir des probabilités

    Args:
        y_hat : Vecteurs one-hot

    Returns:
        Vecteurs de classes (int)
    """
    # Juste un reshape pour avoir une matrice avec une ligne si jamais on a un vecteur des probas d'un seul exemple
    if len(y_hat.shape) == 1:
        y_hat = y_hat.reshape((1,-1))
    
    return  np.argmax(y_hat, axis=1)
    

def score(y, yhat):
    """_summary_
    Args:
        y ():Vecteurs one-hot
        yhat (_type_): Vecteurs des probas de chaque classe
    Returns:
        Accuracy 
    """
    assert (len(y.shape) == 1) or (y.shape[1]==1) , 'La supervision doit etre un vecteur de classes (entiers)'
    assert (len(yhat.shape) == 2) , 'Les prédictions doivent être les vecteurs de probabilités des classes pour chaque exemples'
    
    predictions = pred_classes(yhat)
    nb_bonnes_reponses = np.sum(np.where((y-predictions)==0, 1, 0))

    return  nb_bonnes_reponses/len(predictions) #s, s/len(yhat)


path_train = "dataset/USPS_train.txt"
path_test = "dataset/USPS_test.txt"

trainx, train_y = tools.load_usps(path_train)
testx, test_y = tools.load_usps(path_test)

scaler = StandardScaler()
scaler.fit(trainx)
trainx = scaler.transform(trainx)
testx = scaler.transform(testx)
del scaler
ic.disable()
representations_latentes = np.loadtxt('representations_latentes4.txt')

X_train, X_test, y_train, y_test = train_test_split(representations_latentes, train_y[:len(representations_latentes)], test_size=0.2, random_state=42)
y_train_one_hot = transform_one_hot(y_train)
y_test_one_hot = transform_one_hot(y_test)
loss_celogsoftmax = CELogSoftMax()
softmax = SoftMax()


net2 = Sequentiel(Linear(64, 80, 'lin1'), ReLU(), Linear(80, 10, 'lin2'), softmax)

net2, couts2, opt2 = SGD(net2, X_train, y_train_one_hot, nb_batch=20, loss=loss_celogsoftmax, nb_epochs=10, eps=1e-2, shuffle=True)



raw_scores_train = net2.forward(X_train)
raw_scores_test = net2.forward(X_test)


print("accuracy train: ", score(y_train, raw_scores_train))
print("accuracy test: ", score(y_test, raw_scores_test))
plt.plot(np.arange(len(couts2)), couts2)

