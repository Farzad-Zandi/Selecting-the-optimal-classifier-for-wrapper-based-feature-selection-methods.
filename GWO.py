## Farzad Zandi, 2023.
#  Feature selection by Gray wolf Optimization algorithm.
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
print("feature selection by gray wolf optimization algorithm...")
print("Loading data...")
data = pd.read_csv('D:/Temp/new/data/leukemia.csv', header= None)
N = data.shape[1]-1
label = data.iloc[:,N]
data = data.drop(data.columns[N], axis=1)
# data = data.drop(data.columns[0], axis=1)
print("Data Dimension: ", data.shape)
print("Normalizing Data...")
data = preprocessing.minmax_scale(data, feature_range=(0,1))

[M, N] = data.shape # Data Dimension.
alphaPosition = np.zeros([1, N])
alphaScore = -np.inf
betaPosition = np.zeros([1, N])
betaScore = -np.inf
gammaPosition = np.zeros([1, N])
gammaScore = -np.inf
maxIter = 50 # amximum iteration.
nWolfs = 10 # number of particles.
X = np.random.randint(2, size=(nWolfs, N)) # Initializing random positions.
X = X.astype(np.float64)
Fit = np.full(nWolfs, -np.inf) # Local fitness.
xNew = np.zeros([1, N])
clfs = ['KNN', 'SVM', 'Bayes', 'Ridge', 'DT', 'RF', 'BaggingClassifier', 'LightGBM', 'Perceptron', ' LinearDiscriminantAnalysis'] # Classifiers.
clfProbs = np.zeros(len(clfs))
clfSelection = np.zeros([maxIter+1, len(clfs)])
probs = np.zeros([maxIter+1, len(clfs)]) # Initializing velocities.
bestClf = []
FitLocal = np.full(nWolfs, -np.inf) # Local fitness.

def rouletteWheel(clfProbs, Fit, clfS):
    idx = np.argsort(Fit)
    for i in idx:
        if Fit[int(i)] == -np.inf:
            clfProbs[int(i)] = 0
        else:
            clfProbs[int(clfS[i])] = clfProbs[int(clfS[i])] + Fit[i] / np.sum(Fit) 
    return clfProbs

def cost(x, clfProbs, i):
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    if clfProbs[0] == 0:
        idxData = np.where(np.array(x)==1)[0]
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
        idxData = np.where(np.array(x)==1)[1]
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
            model = LogisticRegression()
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
    model = KNeighborsClassifier(n_neighbors=7)
    idxData = np.where(np.array(x)==1)[0]
    acc = cross_val_score(model, data[:,idxData], label, scoring='accuracy', cv=cv)
    return np.mean(acc)

for i in range(nWolfs):
    # acc, clf = cost(X[i,:], clfProbs,i)
    acc = cost1(X[i,:])
    FitLocal[i] = acc
    # clfSelection[0,i] = clf
    Fit[i] = acc
    if acc>alphaScore:
        alphaScore = acc
        alphaPosition = X[i,:]
        # bestClf = clfs[int(clfSelection[0,np.argmax(FitLocal)])]
    if acc<alphaScore and acc>betaScore:
        betaScore = acc
        betaPosition = X[i,:]
    if acc<alphaScore and acc<betaScore and acc>gammaScore:
        gammaScore = acc
        gammaPosition = X[i,:]
for t in range(maxIter):
    t = t + 1
    a = 2*(1-t/maxIter)
    # clfProbs = rouletteWheel(clfProbs, FitLocal, clfSelection[t-1])
    for i in range(nWolfs):
        for j in range(N):
            r1 = np.random.random()
            r2 = np.random.random()
            r3 = np.random.random()
            A1 = 2*a*r1-a
            C1 = 2*r1
            dAlpha = abs(C1*alphaPosition[j] - X[i,j])
            X1 = alphaPosition[j] - A1*dAlpha
            A2 = 2*a*r2-a
            C2 = 2*r2
            dBeta = abs(C2*betaPosition[j] - X[i,j])
            X2 = betaPosition[j] - A2*dBeta
            A3 = 2*a*r3-a
            C3 = 2*r3
            dGamma = abs(C3*gammaPosition[j] - X[i,j])
            X3 = gammaPosition[j] - A3*dGamma
            xNew[0,j] = (X1 + X2 + X3) / 3
        for j in range(N):
                if np.random.random()< 1/(1+math.exp(-xNew[0,j])):
                    xNew[0,j] = 1
                else:
                    xNew[0,j] = 0
        # clfProbs = rouletteWheel(clfProbs, FitLocal, clfSelection[t-1])
        # acc, clf = cost(xNew, clfProbs,i+1)
        acc = cost1(xNew)
        FitLocal[i] = acc
        # clfSelection[t,i] = clf
        if acc>Fit[i]:
            X[i,:] = xNew
            Fit[i] = acc
    idx = np.argsort(Fit)
    idx = idx[::-1]
    if alphaScore < Fit[idx[0]]:
        alphaScore = Fit[idx[0]]
        alphaPosition = X[idx[0]]
        # bestClf = clfs[int(clfSelection[t,np.argmax(FitLocal)])]
    if betaScore < Fit[idx[1]]:
        betaScore = Fit[idx[1]]
        betaPosition = X[idx[1]]    
    if gammaScore < Fit[idx[2]]:
        gammaScore = Fit[idx[2]]
        gammaPosition = X[idx[2]]
    # print(f"Accuracy in iteration {t} is {np.round(max(FitLocal)*100,2)}, by {clfs[int(clfSelection[t,np.argmax(FitLocal)])]} classiffier.")
    print(f'Accuracy in iteration {t} is: {np.round(alphaScore*100,2)}')
    # probs[t] = clfProbs

idx = np.where(np.array(alphaPosition)==1)[0]
results = pd.DataFrame(idx)
results[' Best Global'] = alphaScore
results.to_csv("D:/Temp/new/Results/GWO/Leukemia/knn/knn10_leukemia_WGO5010.csv")