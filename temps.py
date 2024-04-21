import numpy as np
import matplotlib.pyplot as plt

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

from utils import tools

from icecream import ic



digits = load_digits()

plt.imshow(digits.data[0].reshape(8,8), cmap='grey')
plt.show()
plt.imshow(digits.data[1].reshape(8,8), cmap='grey')

plt.show()

quit()
path_train = "dataset/USPS_train.txt"
path_test = "dataset/USPS_test.txt"

trainx, train_y = tools.load_usps(path_train)
testx, test_y = tools.load_usps(path_test)

ic(trainx.shape)
np.random.seed(0)

scaler = StandardScaler()
scaler.fit(trainx)
trainx = scaler.transform(trainx)
testx = scaler.transform(testx)
del scaler


trainx = trainx[:1000,:]
loss_mse = BCELoss()
lineaire1 = Linear(trainx.shape[1], 64, init_type=1)
lineaire2 = Linear(64, 10, init_type=1)
lineaire3 = Linear(10, 64, init_type=1)
lineaire4 = Linear(64, trainx.shape[1], init_type=1)
sig = Sigmoide()
tanh = Tanh()
tanh2 = Tanh()
tanh3 = Tanh()

iter=100

net = AutoEncodeur(lineaire1, tanh, lineaire2, tanh2, lineaire3, tanh3, lineaire4, sig)
#net = Sequentiel(lineaire1, tanh, lineaire4, sig)
#opt = Optim(net, loss_mse)

net, couts, opt = SGD(net, trainx, trainx,nb_batch=10, loss=loss_mse, nb_epochs=iter, eps=1e-2, shuffle=True)

a = net.get_representation_latente(trainx[0].reshape((1,256)))

print(a)

