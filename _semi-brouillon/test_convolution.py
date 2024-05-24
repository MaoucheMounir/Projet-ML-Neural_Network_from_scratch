import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle

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
from Loss.CELoss import CELoss

from Convolution.convolution1D import *

from utils import tools
from Encapsulation import fonctions as fn

from icecream import ic


def onehot(y):
    onehot = np.zeros((y.size,y.max()+1))
    onehot[np.arange(y.size),y]=1
    return onehot

# Load Data From USPS , directement pris depuis TME4
uspsdatatrain = "C:/_TME\Projet-ML/dataset/USPS_train.txt"
uspsdatatest = "C:/_TME\Projet-ML/dataset/USPS_test.txt" 

alltrainx, alltrainy = tools.load_usps(uspsdatatrain)
alltestx, alltesty = tools.load_usps(uspsdatatest)
alltrainx, alltrainy = alltrainx[:1000],alltrainy[:1000]
alltestx, alltesty = alltestx[:500],alltesty[:500]
# taille couche
input = len(alltrainx[0])
out = len(np.unique(alltesty))
alltrainy = onehot(alltrainy)
alltesty= onehot(alltesty)
alltrainx = alltrainx.reshape(alltrainx.shape[0], alltrainx.shape[1], 1)
alltestx = alltestx.reshape(alltestx.shape[0], alltestx.shape[1], 1)



####################


def pred(x):
    #retourner la meuilleur classe
    return np.argmax(x,axis=1)

#score = []
iteration = 25
gradient_step = 1e-4
batch_size = 100

net = Sequentiel(*[Conv1D(3, 1, 32, stride=1),
                      MaxPool1D(2, 2),
                      Flatten(),
                      Linear(4064, 100, init_type=1),
                      ReLU(),
                      Linear(100, 10,init_type=1),
                      SoftMax()
                        ]) #,label=pred

loss_ce = CELoss()

#optim1 = SGD(net, CELoss(), eps=1e-3)        

net, couts, opt = SGD(net, alltrainx, alltrainy, nb_batch=50, loss=loss_ce, nb_epochs=iteration,shuffle=True)

plt.plot(np.arange(len(couts)),couts,color='red',label="la function de cout")
plt.xlabel("iter")
plt.title("variation de la fonction de cout")
plt.savefig("loss_convolution.png")
#plt.show()

# SGD(net, X_train_scaled, X_train_scaled,nb_batch=10, loss=loss_bce, nb_epochs=iter, eps=1e-1, shuffle=True)
#############################

# with open('net_trained.pkl', 'wb') as f:
#     pickle.dump(net, f)




##################

# alltrainx, alltrainy = tools.load_usps(uspsdatatrain)
# alltestx, alltesty = tools.load_usps(uspsdatatest)
# alltrainx, alltrainy = alltrainx[:5000],alltrainy[:5000]
# alltestx, alltesty = alltestx[:5000],alltesty[:5000]
# alltrainx = alltrainx.reshape(alltrainx.shape[0], alltrainx.shape[1], 1)
# alltestx = alltestx.reshape(alltestx.shape[0], alltestx.shape[1], 1)

# with open('net_trained.pkl', 'rb') as f:
#     net=pickle.load(f)

# # ic(alltrainx.shape)
# # ic(alltrainy.shape)
del opt

_, alltrainy = tools.load_usps(uspsdatatrain)
_, alltesty = tools.load_usps(uspsdatatest)

alltrainy = alltrainy[:1000]
alltesty = alltesty[:500]

predict = net.forward(alltrainx)
# print((np.where(predict == alltrainy, 1, 0)).mean() )
score_train = fn.score(alltrainy,predict)
print("Le score d'accuracy en train = ", score_train)

predict = net.forward(alltestx)
score_test = fn.score(alltesty,predict)
print("Le score d'accuracy en train = ", score_test)

