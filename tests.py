import numpy as np

from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from Lineaire.Linear import *
from Lineaire.MSELoss import *
from NonLineaire.Tanh import Tanh
from NonLineaire.Sigmoide import Sigmoide
from Encapsulation.Sequentiel import Sequentiel
from Encapsulation.Optim import Optim
from Encapsulation.fonctions import SGD

from icecream import ic

# Load the cancer dataset
cancer_X, cancer_y = datasets.load_breast_cancer(return_X_y=True)

# Données de forme (569 exemples, 30 dimensions)
# On récupère une seule dimensions (la N°3), newaxis est fait pour qu'on ait pas un vecteur mais une matrice 442*1
cancer_X = np.array(cancer_X[:, :2])
cancer_y = np.array(cancer_y)

# Split the data into training/testing sets
cancer_X_train = cancer_X[:-20]
cancer_X_test = cancer_X[-20:]

# Split the targets into training/testing sets
cancer_y_train = cancer_y[:-20]
cancer_y_test = cancer_y[-20:]

dim_input = cancer_X_train.shape[1]
dim_output = 1 # le score de chaque exemple, qui va passer par la sigmoide finale
dim_output1 = dim_input #==2

scaler = StandardScaler()
cancer_X_train = scaler.fit_transform(cancer_X_train)

######################################################################

#dim_input = 2
#dim_output1 = 2
#di_output = 1

# 2 -> 2, 2 -> 1
mse_loss = MSELoss()
#net = Sequentiel(Linear(dim_input, dim_output1, 'lineaire 1'), Tanh(), Linear(dim_output1, dim_output, 'lineaire 2'), Sigmoide())
net = Sequentiel(Linear(2, 2, 'lineaire 1'), Tanh(), Linear(2, 1, 'lineaire 2'), Sigmoide())
net.describe_shape()

nb_elements_reduit = 100
cancer_X_train_reduit = cancer_X_train[:nb_elements_reduit]
cancer_y_train_reduit = cancer_y_train[:nb_elements_reduit]

############

optim = Optim(net, mse_loss, eps=10)

for i in range(100):
    optim.step(cancer_X_train_reduit, cancer_y_train_reduit)
    if i % 10 == 0:
        ic(optim.score(cancer_X_train_reduit, cancer_y_train_reduit))
        optim._net.describe_values()


#SGD(net, cancer_X_train_reduit, cancer_y_train_reduit, taille_batch=20, loss=mse_loss)