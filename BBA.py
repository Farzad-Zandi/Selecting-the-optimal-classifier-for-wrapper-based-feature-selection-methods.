## Farzad Zandi, 2023.
#  Feature selection by binary Bat algorithm.
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as LightGBM
from sklearn.linear_model._ridge import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.ensemble._bagging import BaggingClassifier

print("===================================================")
print("Farzad Zandi, 2023...")
print("feature selection by binary bat algorithm...")
print("Loading data...")
data = pd.read_csv('D:/Temp/new/data/prostate.csv', header= None)
N = data.shape[1]-1
label = data.iloc[:,N]
data = data.drop(data.columns[N], axis=1)
# data = data.drop(data.columns[0], axis=1)
print("Data Dimension: ", data.shape)
print("Normalizing Data...")
data = preprocessing.minmax_scale(data, feature_range=(0,1))

def rouletteWheel(clfProbs, Fit, clfS):
    idx = np.argsort(Fit)
    for i in idx:
        if Fit[int(i)] == -np.inf:
            clfProbs[int(i)] = 0
        else:
            clfProbs[int(clfS[i])] = clfProbs[int(clfS[i])] + Fit[i] / np.sum(Fit) 
    return clfProbs

def cost(x, clfProbs, i):
    idxData = np.where(np.array(x)==1)[0]
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    if clfProbs[0] == 0:
        if i==0:
            model = KNeighborsClassifier(n_neighbors=7)
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        if i==1:
            model = svm.SVC()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        if i==2:
            model = GaussianNB()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        if i==3:
            # model = LogisticRegression()
            model = RidgeClassifier()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        if i==4:
            model = DecisionTreeClassifier()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        if i==5:
            model = RandomForestClassifier()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        if i==6:
            # model = MLPClassifier()
            model = BaggingClassifier()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        if i==7:
            model = LightGBM.LGBMClassifier()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        if i==8:
            # model = SnapBoostClassifier()
            model = Perceptron()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        if i==9:
            model = LinearDiscriminantAnalysis()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        clf = i
    else:
        rand = np.random.uniform(np.min(clfProbs), np.max(clfProbs))
        idx = np.argsort(clfProbs)
        if rand<clfProbs[idx[0]]:
            clf = idx[0]
        elif rand<clfProbs[idx[1]]:
            clf = idx[1]
        elif rand<clfProbs[idx[2]]:
            clf = idx[2]
        elif rand<clfProbs[idx[3]]:
            clf = idx[3]
        elif rand<clfProbs[idx[4]]:
            clf = idx[4]
        elif rand<clfProbs[idx[5]]:
            clf = idx[5]
        elif rand<clfProbs[idx[6]]:
            clf = idx[6]
        elif rand<clfProbs[idx[7]]:
            clf = idx[7]
        elif rand<clfProbs[idx[8]]:
            clf = idx[8]
        elif rand<clfProbs[idx[9]]:
            clf = idx[9]
        if clf == 0:
            model = KNeighborsClassifier(n_neighbors=7)
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        elif clf == 1:
            model = svm.SVC()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        elif clf == 2:
            model = GaussianNB()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        elif clf == 3:
            model = RidgeClassifier()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        elif clf == 4:
            model = DecisionTreeClassifier()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        elif clf == 5:
            model = RandomForestClassifier()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        elif clf == 6:
            # model = MLPClassifier()
            model = BaggingClassifier()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        elif clf == 7:
            model = LightGBM.LGBMClassifier()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        elif clf == 8:
            model = Perceptron()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
        elif clf == 9:
            model = LinearDiscriminantAnalysis()
            acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
    return np.mean(acc), clf

def cost1(x):
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    model = BaggingClassifier()
    idxData = np.where(np.array(x)==1)[0]
    acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
    return np.mean(acc)

nFit = np.zeros(10)
res = np.zeros(10)

for h in range(10):
    [M, N] = data.shape # Data Dimension.
    nBats = 10 # Number of bats.
    X = np.random.randint(2, size=(M, N)) # Initializing random positions.
    X = X.astype(np.float64)
    v = np.zeros([M, N]) # Initializing velocities.
    A = 1 + np.random.random(nBats) # Initializing loudness.
    r = np.random.random(nBats) # Initializing random pulse rate.
    Fit = np.full(nBats, -np.inf) # Local fitness.
    FitLocal = np.full(nBats, -np.inf) # Local fitness.
    gFit = -np.inf # Global fitness.
    maxIter = 50 # Maximum iteration.
    alpha = 0.9 # ad-hoc constant.
    gamma = 0.9 # ad-hoc constant.
    r0 = 0.9 # Intializing pulse rate.
    fmin = 0 # Minimum frequency.
    fmax = 1 # Maximum frequency.
    clfs = ['KNN', 'SVM', 'Bayes', 'Ridge', 'DT', 'RF', 'BaggingClassifier', 'LightGBM', 'Perceptron', ' LinearDiscriminantAnalysis'] # Classifiers.
    clfProbs = np.zeros(len(clfs))
    clfSelection = np.zeros([maxIter, len(clfs)])
    probs = np.zeros([maxIter, len(clfs)]) # Initializing velocities.
    bestClf = []
    for t in range(maxIter):
        # clfProbs = rouletteWheel(clfProbs, FitLocal, clfSelection[t-1])
        for i in range(nBats):
            # acc, clf = cost(X[i,:], clfProbs,i)

            acc = cost1(X[i,:])
            FitLocal[i] = acc
            # clfSelection[t,i] = clf
            if np.random.random()<A[i] and acc>Fit[i]:
                Fit[i] = acc
                A[i] = alpha*A[i]
                r[i] = r0*(1-math.exp(-gamma*(t+1)))
        maxFit = np.max(Fit)
        idx = Fit.argmax()
        if maxFit>gFit:
            gFit = maxFit
            # bestClf = clfs[int(clfSelection[t,np.argmax(FitLocal)])]
            xHat = X[idx,:]
        for i in range(nBats):
            beta = np.random.random()
            epsilon = (2*np.random.random()) - 1
            if np.random.random()>r[i]:
                for j in range(N):
                    X[i,j] = X[i,j] + epsilon*np.mean(A)
                    if np.random.random()< 1/(1+math.exp(-X[i,j])):
                        X[i,j] = 1
                    else:
                        X[i,j] = 0
            if np.random.random()<A[i] and Fit[i]<gFit:
                for j in range(N):
                    f = fmin + (fmax-fmin)*beta
                    v[i,j] = v[i,j] + (xHat[j] - X[i,j])*f
                    X[i,j] = X[i,j] + v[i,j]
                    if np.random.random() < 1/(1+math.exp(-X[i,j])):
                        X[i,j] = 1
                    else:
                        X[i,j] = 0
        if gFit==100.0:
            t = maxIter
        # print(f"Accuracy in iteration {t+1} is {np.round(max(FitLocal)*100,2)}, by {clfs[int(clfSelection[t,np.argmax(FitLocal)])]} classiffier.")
        print(f'Accuracy in iteration {t+1} is: {np.round(gFit*100,2)}')

        # probs[t] = clfProbs
    # print(f"Global accuracy: {gFit}")
    # print(f"Best positions : {xHat}")
    idx = np.where(np.array(xHat)==1)[0]
    nFit[h] = len(idx)
    res[h] = gFit

results = pd.DataFrame(nFit)
results['ACC'] = res

# results['Best Classifier'] = bestClf
# probs = pd.DataFrame(probs)
# probs.to_csv("D:/Temp/new/Results/BBA/prostate/probs10_Prostate_BBA5010.csv")

results.to_csv("D:/Temp/new/Results/BBA/prostate/clf/Bagging_prostate_BBA5010.csv")