import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from Lineaire.Linear import *
from Loss.BCELoss import BCELoss
from Activation.Tanh import Tanh
from Activation.Sigmoide import Sigmoide
from Encapsulation.AutoEncodeur import AutoEncodeur
from Encapsulation.Optim import SGD

from utils import tools

from icecream import ic

np.random.seed(5)


digits = load_digits()
#print(digits.data.shape)
# Divisez les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


loss_bce = BCELoss()
lineaire1 = Linear(X_train.shape[1], 32, init_type=1)
lineaire2 = Linear(32, 8, init_type=1)
lineaire3 = Linear(8, 32, init_type=1)
lineaire4 = Linear(32, 64, init_type=1)
sig = Sigmoide()
tanh = Tanh()
tanh2 = Tanh()
tanh3 = Tanh()

iter=100

net = AutoEncodeur(lineaire1, tanh, lineaire2, tanh2, lineaire3, tanh3, lineaire4, sig)
#net = Sequentiel(lineaire1, tanh, lineaire4, sig)

net, couts, opt = SGD(net, X_train, X_train,nb_batch=10, loss=loss_bce, nb_epochs=iter, eps=1e-1, shuffle=True)

i = 0
def predict(i, X, Y):
    dim = int(np.sqrt(X.shape[1]))
    
    plt.figure(figsize=(5, 2))  # Ajustez la taille de la figure selon vos besoins
    #plt.title(f'classe = {Y[i]}')
    
    plt.subplot(1, 2, 1)  # Première cellule de la grille
    plt.imshow(X[i].reshape((dim,dim)), cmap='grey')
    plt.title(f'Image originale ({Y[i]})')
    pred = opt._net.forward(X[i].reshape((1,dim**2)))
    plt.subplot(1, 2, 2)  # Deuxième cellule de la grille
    plt.imshow(pred.reshape((dim,dim)), cmap='grey')
    plt.title('Image reconstruite')
    plt.show()
    #print('classe = ', y_train[i])

for i in range(10):
    predict(i, X_train, y_train)

plt.plot(range(len(couts)), couts)
plt.show()


quit()

def pred(raw_scores):
    return np.where(raw_scores>=0.5, 1, 0)

couts = opt._couts
print("accuracy : ", opt.score(pred(net.forward(X_train)),X_train))

fig,ax = plt.subplots(ncols=3,nrows=1,figsize=(20,5))
ax.flatten()

tools.plot_frontiere(X_train,opt._net.predict,ax=ax[0])
tools.plot_data(X_train, X_train,ax[0])
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")
ax[0].set_title(" Frontière de décison en apprentissage")

# tools.plot_frontiere(testx,opt._net.predict,ax=ax[1])
# tools.plot_data(testx, testy,ax[1])
# ax[1].set_xlabel("x1")
# ax[1].set_ylabel("x2")
# ax[1].set_title(" Frontière de décison en test")


ax[2].plot(np.arange(len(couts)),couts,color='red')
#ax[2].plot(np.arange(iter),l_std,color='black')
#ax[2].legend(('Moyenne', 'Variance'))
ax[2].legend(["Cout"])
ax[2].set_xlabel("Nombre d'epochs")
ax[2].set_ylabel("MSE")
ax[2].set_title("Variation de la MSE")
plt.show()

