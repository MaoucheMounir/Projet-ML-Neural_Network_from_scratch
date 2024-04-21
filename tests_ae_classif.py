import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Lineaire.Linear import *
from Loss.BCELoss import BCELoss
from Activation.Tanh import Tanh
from Activation.Sigmoide import Sigmoide
from Encapsulation.AutoEncodeur import AutoEncodeur
from Encapsulation.Sequentiel import Sequentiel
from Encapsulation.Optim import SGD
from Activation.SoftMax import  SoftMax
from Loss.CELogSoftMax import CELogSoftMax

from utils import tools

from icecream import ic

path_train = "dataset/USPS_train.txt"
trainx, train_y = tools.load_usps(path_train)
train_y = train_y[:1000]

representations_latentes = np.loadtxt('representations_latentes.txt')
ic(representations_latentes.shape)
ic(train_y.shape)
X_train, X_test, y_train, y_test = train_test_split(representations_latentes, train_y, test_size=0.2, random_state=42)


def transform_one_hot(classe):
    a =  np.zeros(10)
    a[classe] = 1
    return a

y_train_one_hot=[]
for y in y_train:
    y_train_one_hot.append(transform_one_hot(y))
    
y_train_one_hot = np.array(y_train_one_hot)

#ic(X_train.shape, y_train_one_hot.shape) (800,10), (800,10)


loss_celogsoftmax = CELogSoftMax()
softmax = SoftMax()

net2 = Sequentiel(Linear(10, 5, 'lin1'), Tanh(), Linear(5, 10, 'lin2'), softmax)

net2, couts2, opt2 = SGD(net2, X_train, y_train_one_hot, nb_batch=20, loss=loss_celogsoftmax, nb_epochs=500, eps=1e-2, shuffle=False)

raw_scores = net2.forward(X_train)

def pred_classes(y_hat):
    classes_predites = np.argmax(y_hat, axis=1)
    predictions = []
    for y in classes_predites:
        predictions.append(transform_one_hot(y))
    
    predictions= np.array(predictions)
    return predictions

def score(y, yhat):
    predictions = pred_classes(yhat)
    ic(predictions)
    ic(y)
    comparaison = ic((predictions == 1) * (y == 1))
    s = np.sum(comparaison)
    return s, s/len(yhat)

print("accuracy : ", score(y_train_one_hot, raw_scores))
