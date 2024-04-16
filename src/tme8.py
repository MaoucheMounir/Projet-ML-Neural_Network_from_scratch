"""
MAOUCHE Mounir
TME 8 
"""
import numpy as np
import matplotlib.pyplot as plt

##Generation de donnees;

def gen_data_lin(a, b, sig, N=500, Ntest=1000):
    X_train = np.sort(np.random.rand(N)) 
    X_test  = np.sort(np.random.rand(Ntest))
    y_train = a * X_train + b + np.random.normal(0, sig, N)
    y_test  = a * X_test + b + np.random.normal(0, sig, Ntest)
    return X_train, y_train, X_test, y_test

##Validation des formules analytiques;

###Avec Variance et covaraiance
def modele_lin_analytique0(X_train, Y_train):
    cov_xy = np.cov(X_train, Y_train)
    var_x = np.var(X_train)
    
    a_hat = cov_xy /var_x
    
    esp_x = np.mean(X_train)
    esp_y = np.mean(Y_train)

    b_hat = esp_y - a_hat * esp_x  

    return a_hat, b_hat

def modele_lin_analytique(X_train, y_train):
    cov = np.cov(X_train, y_train)
    ahat = cov[0,1] / cov[0,0]
    bhat = y_train.mean() - ahat * X_train.mean()
    return ahat, bhat

def calcul_prediction_lin(X_train, ahat, bhat):
    y_hat_train = ahat* X_train + bhat 

    return y_hat_train

def erreur_mc(y_train, yhat_train):
    e = yhat_train - y_train
    e_T = np.transpose(e)

    return np.mean(e_T * e)


def dessine_reg_lin(X_train, y_train, X_test, y_test, a, b):
    plt.figure()
    plt.plot(X_test, y_test, 'r.')
    plt.plot(X_train, y_train, 'b')

    
    yhat_test = calcul_prediction_lin(X_test, a, b)

    plt.plot(X_test, yhat_test, 'g', lw=3)

##Formulation au sens des moindres carres

def make_mat_lin_biais(X_train):
    d = X_train.shape[0]
    uns = np.full((d,1), 1)
    return (np.hstack((X_train.reshape(d,1), uns)))

def reglin_matriciel(Xe,y_train):
    X_T = np.transpose(Xe)
    
    A = np.matmul(X_T, Xe)
    B = np.matmul(X_T, y_train)

    w = np.linalg.solve(A, B)

    return w
    
##donnees Polynomiales

def gen_data_poly2(a, b, c, sig, N=500, Ntest=1000):
    '''
    Tire N points X aléatoirement entre 0 et 1 et génère y = ax^2 + bx + c + eps
    eps ~ N(0, sig^2)
    '''
    X_train = np.sort(np.random.rand(N))
    X_test  = np.sort(np.random.rand(Ntest))
    y_train = a*X_train**2+b*X_train+c+np.random.randn(N)*sig
    y_test  = a*X_test**2 +b*X_test +c+np.random.randn(Ntest)*sig
    return X_train, y_train, X_test, y_test


def make_mat_poly_biais(X): # fonctionne pour un vecteur unidimensionel X
    N = len(X)
    uns = np.ones((N,1))
    return np.hstack((X.reshape(N,1)**2, X.reshape(N,1), uns))


def calcul_prediction_matriciel(Xe, w):
    y_hat = (w * Xe).sum(axis=1)
    return y_hat

def dessine_poly_matriciel(Xp_train,yp_train,Xp_test,yp_test,w):
    plt.figure()
    plt.plot(Xp_test, yp_test, 'r.')
    plt.plot(Xp_train, yp_train, 'b')

    
    yhat_test = calcul_prediction_matriciel(Xp_test, w)

    plt.plot(Xp_test, yhat_test, 'g', lw=3)


##fct cout et Descente de gradient
"""def descente_grad_mc(Xe, y_train, eps=1e-4, nIterations=500):
    N = Xe.shape[0]
    w = np.zeros((N,1))
    
    def cout(w_t):


    for i in range(nIterations):
        #w[i] = w[i-1] - eps*grad(cout(w[i]))

"""

def descente_grad_mc(X, y, eps=1e-4, nIterations=100):
    w = np.zeros(X.shape[1]) # init à 0
    allw = [w]
    for _ in range(nIterations):
        w = w - eps * 2 * X.T @ (X @ w - y)
        allw.append(w) # stockage de toutes les valeurs intermédiaires pour analyse
    allw = np.array(allw)
    return w, allw # la dernière valeur (meilleure) + tout l'historique pour le plot

def application_reelle(X_train,y_train,X_test,y_test):
    w = np.linalg.solve(X_train.T@X_train,X_train.T@y_train)
    yhat   = np.dot(w,X_train.T)
    yhat_t = np.dot(w,X_test.T)
    return w, yhat, yhat_t

##Normalisation

def normalisation0(X_train, X_test):
    '''
    Fonction de normalisation des données pour rendre les colonnes comparables
    Chaque variable sera assimilée à une loi normale qu'il faut centrer + réduire.
    ATTENTION: il faut calculer les moyennes et écarts-types sur les données d'apprentissage seulement
    '''
    # A compléter
    # 1) calcul des moyennes et écarts types pour chaque colonne
    # 2) normalisation des colonnes
    # 3) Ajout d'un biais: fourni ci-dessous)
    mu = X_train.mean(axis=0)
    sig = X_train.std(axis=0)
    Xn_train = (X_train - mu) / sig
    Xn_test = (X_test - mu) / sig
    Xn_train = np.hstack((Xn_train, np.ones((Xn_train.shape[0], 1))))
    Xn_test   = np.hstack((Xn_test, np.ones((Xn_test.shape[0], 1))))
    return Xn_train, Xn_test

def normalisation(X_train, X_test):
    '''
    Fonction de normalisation des données pour rendre les colonnes comparables
    Chaque variable sera assimilée à une loi normale qu'il faut centrer + réduire.
    ATTENTION: il faut calculer les moyennes et écarts-types sur les données d'apprentissage seulement
    '''
    means_stds = list(zip(X_train.mean(axis=0),X_train.std(axis=0)))
    Xn_train  = np.apply_along_axis(lambda x:np.array(\
                [ (x[i]-means_stds[i][0])/means_stds[i][1] for i in range(len(x))]) , 1 ,X_train)
    Xn_test   = np.apply_along_axis(lambda x:np.array(\
                [ (x[i]-means_stds[i][0])/means_stds[i][1] for i in range(len(x))]) , 1 ,X_test)
    
    Xn_train = np.hstack((Xn_train, np.ones((Xn_train.shape[0], 1))))
    Xn_test   = np.hstack((Xn_test, np.ones((X_test.shape[0], 1))))
    return Xn_train, Xn_test


def plot_y(y_train, y_test, yhat, yhat_t):
    # tracé des prédictions:
    fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=2, sharex='col', subplot_kw=dict(frameon=False)) 
    
    ax1[0].set_title('En test')
    ax1[0].plot(y_test, label="GT")
    ax1[0].plot(yhat_t, label="pred")
    ax1[0].legend()
    
    ax1[1].set_title('En train')
    ax1[1].plot(y_train, label="GT")
    ax1[1].plot(yhat, label="pred")
    
    ax2[0].set_title('$sorted(|err|)$ en test')
    ax2[0].plot(sorted(abs(y_test-yhat_t)), label="diff")
    
    ax2[1].set_title('$sorted(|err|)$ en train')
    ax2[1].plot(sorted(abs(y_train-yhat)), label="diff")
    return

## Ouverture
###Selection des caracteristiques
def eval_MC_sans_vars(X_train, y_train, X_test, y_test):
    Xnt_train, Xnt_test = normalisation(X_train[:, [3,5]], X_test[:, [3,5]])
    w = reglin_matriciel(Xnt_train, y_train)

    yhat   = Xnt_train @ w
    yhat_t = Xnt_test @ w
    print('Erreur moyenne au sens des moindres carrés (train):', erreur_mc(yhat, y_train))
    print('Erreur moyenne au sens des moindres carrés (test):', erreur_mc(yhat_t, y_test))
    plot_y(y_train, y_test, yhat, yhat_t)

    plt.figure()
    plt.bar(np.arange(len(w)), w)

## Encodage de l'origine
### One hot encoding

def create_index(unique_cat):
    dict_index = dict()
    for i, cat in enumerate(unique_cat):
        dict_index[cat] = i
    return dict_index

def one_hot_encoding(X):
    unique_cat = np.unique(X)
    dict_index = create_index(unique_cat)
    ohe_X = np.zeros((len(X), len(dict_index)))
    for i, cat in enumerate(X):
        ohe_X[i, dict_index[cat]] = 1
    return ohe_X

def encoder_origine(data):
    origine = data.values[:, -2]
    ohe_origin = one_hot_encoding(origine)
    #on pourrait afficher le plot ici aussi mais ce serait incomprehensible tellement il y a de valeurs
    #data[[7, 8]].dropna().groupby(by=8).mean().plot(kind='bar') 
    
    return ohe_origin

##Encodage de l'annee
def encoder_annee(data):
    years = data.values[:, -3]
    np.unique(years, return_counts=True), 
    data[[0, 6]].groupby(by=6).mean().plot(kind='bar')
    #Comme on peut le voir dans le plot, il y a 13 annees et les valeurs sont assez equilibrees donc il est difficile de reduire le nombre de dimensions (fusionner des annees) sans trop desequilibrer les valeurs
    ohe_years = one_hot_encoding(years)

    return ohe_years

def separation_train_test(X, y, pc_train=0.75):
    index = np.arange(len(y))
    np.random.shuffle(index) # liste mélangée
    napp = int(len(y)*pc_train)
    X_train, y_train = X[index[:napp]], y[index[:napp]]
    X_test, y_test   = X[index[napp:]], y[index[napp:]]
    return X_train, y_train, X_test, y_test


def eval_MC_avec_one_hot_encoding(data, ohe_years, ohe_origin, y):
    Xoh = np.array(data.values[:, 1:-3], dtype=np.float64) # Sans année et model
    Xoh = np.hstack((Xoh, ohe_years, ohe_origin, np.ones((Xoh.shape[0], 1))))
    Xoh_train, y_train, Xoh_test, y_test = separation_train_test(Xoh, y, pc_train=0.75)

    Xnoh_train, Xnoh_test = normalisation(Xoh_train[:, :-1], Xoh_test[:, :-1]) # Sans le biais car on l'ajoute deux fois sinon
    w = reglin_matriciel(Xnoh_train, y_train)

    yhat   = Xnoh_train @ w
    yhat_t = Xnoh_test @ w
    print('Erreur moyenne au sens des moindres carrés (train):', erreur_mc(yhat, y_train))
    print('Erreur moyenne au sens des moindres carrés (test):', erreur_mc(yhat_t, y_test))
    plt.bar(np.arange(len(w)), w)

##Dans l'application reelle, dans les w il manque une dimension, ils ont oublie W0
##seed(0) pour avoir les random dans le meme sens
##ppour avoir les memes resultats, faire X_train en random(N) et X_test (Ntest) et Y train (N) et Y_test (Ntest)