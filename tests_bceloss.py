from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Lineaire.Linear import Linear
from NonLineaire.Tanh import Tanh
from MultiClasse.CELogSoftMax import CELogSoftMax
from MultiClasse.CELoss import CELoss
from MultiClasse.SoftMax import SoftMax
from Encapsulation.Sequentiel import Sequentiel
from Encapsulation.fonctions import SGD

from icecream import ic


digits = load_digits()
#print(digits.data.shape)
# Divisez les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# plt.imshow(digits.images[64], cmap='grey')
# plt.show()
ic.disable()

def transform_one_hot(classe):
    a =  np.zeros(10)
    a[classe] = 1
    return a

y_train_one_hot=[]
for y in y_train:
    y_train_one_hot.append(transform_one_hot(y))
    
y_train_one_hot=np.array(y_train_one_hot)
#print(y_train_one_hot)

lineaire1 = Linear(64, 8, name='lin1')
tanh = Tanh()
lineaire2 = Linear(8, 10, name='lin2')
loss_celogsoftmax = CELogSoftMax()
softmax = SoftMax()
#loss_CE = CELoss()
softmax = SoftMax()

net = Sequentiel(lineaire1, tanh, lineaire2, softmax)

ic(X_train.shape[1])

net, couts, opt = SGD(net, X_train, y_train_one_hot, nb_batch=20, loss=loss_celogsoftmax, nb_epochs=300, eps=1e-2, shuffle=False)

plt.plot(np.arange(len(couts)), couts)
plt.show()

ic.enable()

raw_scores = net.forward(X_train)
#pred = np.where(pred >= 0.5, 1, 0)
#ic(raw_scores)

# def score(y, yhat):
#     diff = y-yhat
#     bonnes_reponses = np.where(diff >0, 1, 0)
    
#     return np.sum(bonnes_reponses)

def pred_classes(y_hat):
    classes_predites = np.argmax(y_hat, axis=1)
    predictions = []
    for y in classes_predites:
        predictions.append(transform_one_hot(y))
    
    predictions= np.array(predictions)
    return predictions


#yhat = pred_classes(raw_scores)

def score(y, yhat):
    predictions = pred_classes(yhat)
    ic(predictions)
    ic(y)
    comparaison = ic((predictions == 1) * (y == 1))
    s = np.sum(comparaison)
    return s, s/len(yhat)

print("accuracy : ", score(y_train_one_hot, raw_scores))


