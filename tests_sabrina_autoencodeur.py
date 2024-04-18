from Lineaire.Linear import *
from Loss.BCELoss import BCELoss
from Activation.Tanh import Tanh
from Activation.Sigmoide import Sigmoide
from Encapsulation.Sequentiel import Sequentiel
from Encapsulation.Optim import Optim, SGD
from utils import tools

from icecream import ic

path_train = "dataset/USPS_train.txt"
path_test = "dataset/USPS_test.txt"

trainx, train_y = tools.load_usps(path_train)
testx, test_y = tools.load_usps(path_test)

ic(train_y.shape)

# auto encdoder symertique

out1=64
out2=10

BCE_loss = BCELoss()
modul_lin1 = Linear( trainx.shape[1], out1)
modul_lin2 = Linear(out1, out2)
modul_lin3 = Linear( out2, out1)
modul_lin4 = Linear(out1, trainx.shape[1])

modul_lin3._parameters = modul_lin2._parameters.T
modul_lin4._parameters = modul_lin1._parameters.T

encoder=[modul_lin1,Tanh(),modul_lin2,Tanh()]
decoder=[modul_lin3,Tanh(),modul_lin4,Sigmoide()]

encoder_100_10 = Sequentiel(*(encoder+decoder))

#opt = Optim(encoder_100_10,BCE_loss,1e-4)
nb_iter = 100
#l_loss = SGD(trainx,trainx,100,epochs=1000,shuffle=True)
net, couts, opt = SGD(encoder_100_10, trainx, trainx, nb_batch=10, loss=BCE_loss, nb_epochs=nb_iter, eps=1e-3, shuffle=False)

tools.print_auto_encoder(testx,encoder_100_10,16,16)