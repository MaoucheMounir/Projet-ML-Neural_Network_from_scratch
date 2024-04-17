import src.mltools as tl
from src import tools
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
size = 1000

datax, datay = tl.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=size, data_type=1, epsilon=0.1)
testx, testy = tl.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=size, data_type=1, epsilon=0.1)


datay = np.where(datay==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))


loss_mse = MSELoss()
modul_lin1 = Linear( datax.shape[1], 4, init_type=1)
modul_lin2 = Linear(4, 1, init_type=1)
modul_sig = Sigmoide()
modul_tan = Tanh()

l_loss=[]
iter=1500

for _ in range(iter):
    
    #forward
    new_feautre_lin_1 = modul_lin1.forward(datax)   
    new_feautre_activation_1 = modul_tan.forward(new_feautre_lin_1)
    new_feautre_lin_2 = modul_lin2.forward(new_feautre_activation_1)
    y_hat = modul_sig.forward(new_feautre_lin_2)
    loss = loss_mse.forward(datay,y_hat)
    l_loss.append(loss.mean())
    
    #backward
    loss_back = loss_mse.backward(datay, y_hat)
    delta_sig = modul_sig.backward_delta(new_feautre_lin_2,loss_back)
    delta_lin2 = modul_lin2.backward_delta(new_feautre_activation_1,delta_sig)
    delta_tan = modul_tan.backward_delta(new_feautre_lin_1,delta_lin2)
    delta_lin1 = modul_lin1.backward_delta(datax,delta_tan)

    modul_lin2.backward_update_gradient(new_feautre_activation_1, delta_sig)
    modul_lin1.backward_update_gradient(datax, delta_tan)    

    modul_lin2.update_parameters(gradient_step = 1e-3)
    modul_lin1.update_parameters(gradient_step = 1e-3)
    
    #modul_lin1.describe_values()
    #modul_lin1.describe_values()
    
    modul_lin1.zero_grad()
    modul_lin2.zero_grad()
    
    
    


def predict(x):
    hidden_l = modul_lin1.forward(x)
    hidden_l = modul_tan.forward(hidden_l)
    hidden_l = modul_lin2.forward(hidden_l)
    hidden_l = modul_sig.forward(hidden_l)  
    return np.where(hidden_l >= 0.5, 1, 0)

print("accuracy : ",np.where(testy == predict(testx),1,0).mean())

fig,ax=plt.subplots(ncols=3,nrows=1,figsize=(20,5))
ax.flatten()

tools.plot_frontiere(datax,predict,ax=ax[0])
tools.plot_data(datax, datay,ax[0])
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")
ax[0].set_title(" frontiere de decison en app")

tools.plot_frontiere(testx,predict,ax=ax[1])
tools.plot_data(testx, testy,ax[1])
ax[1].set_xlabel("x1")
ax[1].set_ylabel("x2")
ax[1].set_title(" frontiere de decison en test")


ax[2].plot(np.arange(iter),l_loss,color='red',label="la function de cout")
ax[2].set_xlabel("iter")
ax[2].set_ylabel("mse")
ax[2].set_title("variation de la mse")
plt.show()










