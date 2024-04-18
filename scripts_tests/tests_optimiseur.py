import numpy as np

import matplotlib.pyplot as plt

from Lineaire.Linear import *
from Loss.MSELoss import *
from Activation.Tanh import Tanh
from Activation.Sigmoide import Sigmoide
from Encapsulation.Sequentiel import Sequentiel
from Encapsulation.Optim import Optim, SGD
from utils import tools

from icecream import ic

size =1000
np.random.seed(5)

datax, datay = tools.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=size, data_type=1, epsilon=0.1)
testx, testy = tools.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=size, data_type=1, epsilon=0.1)

datay = np.where(datay==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))

loss_mse = MSELoss()
lineaire1 = Linear(datax.shape[1], 4, init_type=1)
lineaire2 = Linear(4, 1, init_type=1)
sig = Sigmoide()
tanh = Tanh()

l_loss=[]
iter=10

net = Sequentiel(lineaire1, tanh, lineaire2, sig)
opt = Optim(net, loss_mse, eps=1e-4)

for _ in range(iter):
    opt.step(datax, datay)

def pred(raw_scores):
    return np.where(raw_scores>=0.5, 1, 0)

couts = opt._couts
print("accuracy : ", opt.score(pred(net.forward(datax)),datay))

fig,ax=plt.subplots(ncols=3,nrows=1,figsize=(20,5))
ax.flatten()

tools.plot_frontiere(datax,opt._net.predict,ax=ax[0])
tools.plot_data(datax, datay,ax[0])
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")
ax[0].set_title(" Frontière de décison en apprentissage")

tools.plot_frontiere(testx,opt._net.predict,ax=ax[1])
tools.plot_data(testx, testy,ax[1])
ax[1].set_xlabel("x1")
ax[1].set_ylabel("x2")
ax[1].set_title(" Frontière de décison en test")


ax[2].plot(np.arange(len(couts)),couts,color='red')
#ax[2].plot(np.arange(iter),l_std,color='black')
#ax[2].legend(('Moyenne', 'Variance'))
ax[2].legend(["Cout"])
ax[2].set_xlabel("Nombre d'epochs")
ax[2].set_ylabel("MSE")
ax[2].set_title("Variation de la MSE")
plt.show()

