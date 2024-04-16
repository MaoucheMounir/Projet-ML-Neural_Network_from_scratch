import src.mltools as tl

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
size = 1500

datax, datay = tl.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=size, data_type=0, epsilon=0.1)
testx, testy = tl.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=size, data_type=0, epsilon=0.1)


datay = np.where(datay==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))


nh = 80
nh2 = 60

def pred(x):
    return np.where(x >= 0.5,1, 0)


loss_mse = MSELoss()
couche_lin = Linear( datax.shape[1], nh)
couche_lin2 = Linear(nh, nh2)
couche_lin3 = Linear(nh2, 1,)

sig = Sigmoide()
tan = Tanh()
ic.enable()
net = Sequentiel(*[couche_lin,tan,couche_lin2,tan,couche_lin3,sig])

#opt = Optim(net,loss_mse,eps=1e-4)
sizeba = 100
net, couts, opt = SGD(net, datax, datay, sizeba, loss_mse, nb_epochs=100, eps=1e-5)

print("accuracy : ", opt.score(testx,testy))


tl.plot_frontiere(testx, opt._net.predict, step=100)
tl.plot_data(testx, testy.reshape(-1))
plt.figure()
plt.plot(list(range(len(couts))),'black')
# plt.plot(std,'blue')
plt.legend(('Moyenne', 'Variance'))
plt.show()