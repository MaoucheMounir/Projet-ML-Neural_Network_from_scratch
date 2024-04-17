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

#np.random.seed()

datay = np.where(datay==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))

loss_mse = MSELoss()
modul_lin1 = Linear( datax.shape[1], 4)
modul_lin2 = Linear(4, 1)
modul_sig = Sigmoide()
modul_tan = Tanh()

l_loss=[]
iter=10000
eps = 1e-4
couts = []

net = Sequentiel(*[modul_lin1,modul_tan,modul_lin2,modul_sig])

for _ in range(iter):
    #net.describe_values()
    output = net.forward(datax) #fait
    cout = loss_mse.forward(datay, output).mean() #fait
    
    gradient_loss = loss_mse.backward(datay, output) # fait, de taille (100,1)
    
    # net.backward_delta(datax, gradient_loss)
    # net.backward_update_gradient(datax, gradient_loss)
    net.backward(datax, gradient_loss)
    
    net.update_parameters(eps)
    couts.append(cout)
    
    
def score(X, Y):
        pred=np.where(net.forward(X)>=0.5,1,0)
        return np.where(Y == pred, 1, 0).mean()


print("accuracy : ", score(datax,datay))

fig,ax=plt.subplots(ncols=3,nrows=1,figsize=(20,5))
ax.flatten()

tools.plot_frontiere(datax,net.predict,ax=ax[0])
tools.plot_data(datax, datay,ax[0])
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")
ax[0].set_title(" frontiere de decison en app")

tools.plot_frontiere(testx, net.predict,ax=ax[1])
tools.plot_data(testx, testy,ax[1])
ax[1].set_xlabel("x1")
ax[1].set_ylabel("x2")
ax[1].set_title(" frontiere de decison en test")


ax[2].plot(np.arange(len(couts)),couts,color='red')
#ax[2].plot(np.arange(iter),l_std,color='black')
ax[2].legend(('Moyenne', 'Variance'))
ax[2].set_xlabel("iter")
ax[2].set_ylabel("mse")
ax[2].set_title("variation de la mse")
plt.show()

