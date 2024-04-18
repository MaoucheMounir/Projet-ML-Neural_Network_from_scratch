"""import numpy as np

# Vecteur numpy
vecteur = np.array([1, 2, 3])

# Nombre de fois que vous voulez dupliquer le vecteur (nombre de colonnes)
nb_colonnes = 5

# Répéter chaque élément du vecteur
repetitions = np.repeat(vecteur[:, np.newaxis], nb_colonnes, axis=1)

print("Matrice résultante :\n", repetitions)
"""

import numpy as np
from MultiClasse.SoftMax import SoftMax
from icecream import ic

scores = np.array([
    [24, 45, 2],
    [180, 1, 15],
    [52, 34, 99],
])

y = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0]
])

softmax = SoftMax()
yhat = softmax.forward(scores)

def transform_one_hot(classe):
    a =  np.zeros(3)
    a[classe] = 1
    return a


def pred_classes(y_hat):
    classes_predites = np.argmax(y_hat, axis=1)
    predictions = []
    for y in classes_predites:
        predictions.append(transform_one_hot(y))
    
    predictions= np.array(predictions)
    return predictions

    
    
pred_classes(yhat)

def score(y, yhat):
    predictions = pred_classes(yhat)
    ic(predictions)
    ic(y)
    comparaison = ic((predictions == 1) * (y == 1))
    
    return np.sum(comparaison)

print("accuracy : ", score(y, yhat))