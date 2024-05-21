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

from convolution import *

from utils import tools
from Encapsulation import fonctions as fn

from icecream import ic


with open('net_trained.pkl', 'rb') as f:
    net=pickle.load(f)

# ic(alltrainx.shape)
# ic(alltrainy.shape)


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


predict = net.forward(alltrainx)
# print((np.where(predict == alltrainy, 1, 0)).mean() )
print("Le score d'accuracy en train = ",fn.score(alltrainy,predict))
predict = net.forward(alltestx)
print("Le score d'accuracy en train = ",fn.score(alltesty,predict))
