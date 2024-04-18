import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from Lineaire.Linear import *
from Loss.MSELoss import *
from Activation.Tanh import Tanh
from Activation.Sigmoide import Sigmoide
from Encapsulation.Sequentiel import Sequentiel
from Encapsulation.Optim import SGD

from src import tools

from icecream import ic

size=1000

np.random.seed(5)

datax, datay = tools.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=size, data_type=1, epsilon=0.1)
testx, testy = tools.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=size, data_type=1, epsilon=0.1)

#np.random.seed()

datay = np.where(datay==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))

loss_mse = MSELoss()
lineaire1 = Linear( datax.shape[1], 4, init_type=1)
lineaire2 = Linear(4, 1, init_type=1)
sig = Sigmoide()
tan = Tanh()

l_loss=[]
iter=100

net= Sequentiel(lineaire1, tan, lineaire2, sig)

net, couts, opt = SGD(net, datax,datay,nb_batch=1, loss=loss_mse, nb_epochs=iter, eps=1e-3, shuffle=False)

pred = np.where(net.forward(datax)>=0.5,1,0)
print("accuracy : ", opt.score(datay, pred))

fig,ax=plt.subplots(ncols=3,nrows=1,figsize=(20,5))
ax.flatten()

tools.plot_frontiere(datax,opt._net.predict,ax=ax[0])
tools.plot_data(datax, datay,ax[0])
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")
ax[0].set_title("Frontière de décison en apprentissage")

tools.plot_frontiere(testx, opt._net.predict, ax=ax[1])
tools.plot_data(testx, testy, ax[1])
ax[1].set_xlabel("x1")
ax[1].set_ylabel("x2")
ax[1].set_title(" frontiere de decison en test")


ax[2].plot(np.arange(len(couts)), couts, color='red')
#ax[2].plot(np.arange(iter),l_std,color='black')
#ax[2].legend(('Moyenne', 'Variance'))
ax[2].legend(["Cout"])
ax[2].set_xlabel("Nombre d'epochs")
ax[2].set_ylabel("MSE")
ax[2].set_title("Variation de la MSE")
plt.show()

