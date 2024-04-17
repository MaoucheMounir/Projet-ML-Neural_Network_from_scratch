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
from src import tools

from icecream import ic

size =1000

np.random.seed(5)

datax, datay = tools.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=size, data_type=1, epsilon=0.1)
testx, testy = tools.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=size, data_type=1, epsilon=0.1)

np.random.seed()

datay = np.where(datay==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))

loss_mse = MSELoss()
modul_lin1 = Linear( datax.shape[1], 4)
modul_lin2 = Linear(4, 1)
modul_sig = Sigmoide()
modul_tan = Tanh()

l_loss=[]
iter=100

net= Sequentiel(*[modul_lin1,modul_tan,modul_lin2,modul_sig])

net, couts, opt = SGD(net, datax,datay,taille_batch=100, loss=loss_mse, nb_epochs=iter, eps=1e-4, shuffle=False)

print("accuracy : ", opt.score(datax,datay))

fig,ax=plt.subplots(ncols=3,nrows=1,figsize=(20,5))
ax.flatten()

tools.plot_frontiere(datax,opt._net.predict,ax=ax[0])
tools.plot_data(datax, datay,ax[0])
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")
ax[0].set_title(" frontiere de decison en app")

tools.plot_frontiere(testx,opt._net.predict,ax=ax[1])
tools.plot_data(testx, testy,ax[1])
ax[1].set_xlabel("x1")
ax[1].set_ylabel("x2")
ax[1].set_title(" frontiere de decison en test")


#ax[2].plot(np.arange(iter),l_mean,color='red')
#ax[2].plot(np.arange(iter),l_std,color='black')
ax[2].legend(('Moyenne', 'Variance'))
ax[2].set_xlabel("iter")
ax[2].set_ylabel("mse")
ax[2].set_title("variation de la mse")
plt.show()
















































# size = 1500

# datax, datay = tl.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=size, data_type=1, epsilon=0.1)
# testx, testy = tl.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=size, data_type=1, epsilon=0.1)


# datay = np.where(datay==-1,0,1).reshape((-1,1))
# testy = np.where(testy==-1,0,1).reshape((-1,1))


# nh = 80
# nh2 = 60

# def pred(x):
#     return np.where(x >= 0.5,1, 0)


# loss_mse = MSELoss()
# couche_lin = Linear( datax.shape[1], nh)
# couche_lin2 = Linear(nh, nh2)
# couche_lin3 = Linear(nh2, 1,)

# sig = Sigmoide()
# tan = Tanh()
# ic.enable()
# net = Sequentiel(*[couche_lin,tan,couche_lin2,tan,couche_lin3,sig])

# #opt = Optim(net,loss_mse,eps=1e-4)
# sizeba = 100
# net, couts, opt = SGD(net, datax, datay, sizeba, loss_mse, nb_epochs=100, eps=1e-4)

# print("accuracy : ", opt.score(testx,testy))


# tl.plot_frontiere(testx, opt._net.predict, step=100)
# tl.plot_data(testx, testy.reshape(-1))
# plt.figure()
# plt.plot(list(range(len(couts))),'black')
# # plt.plot(std,'blue')
# plt.legend(('Moyenne', 'Variance'))
# plt.show()

















