
import pandas as pd
import numpy as np
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS, summarize)
from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

Smarket = load_data('Smarket')

#allvars = Smarket.columns.drop(['Today', 'Direction', 'Year'])
design = MS(['Lag1', 'Lag2'])
X = design.fit_transform(Smarket)
y = Smarket.Direction == 'Up'
train = (Smarket.Year < 2005)
Smarket_train = Smarket.loc[train]
Smarket_test = Smarket.loc[~train]

X_train, X_test = X.loc[train], X.loc[~train]
y_train, y_test = y.loc[train], y.loc[~train]
D = Smarket.Direction
L_train, L_test = D.loc[train], D.loc[~train]

X_train  = X_train.drop(columns=['intercept'])
X_test = X_test.drop(columns=['intercept'])

NB = GaussianNB()
print(X_train[0:5])
print(L_train[0:5])
NB.fit(X_train, L_train)
print("The means of each predictor (columns) for each class (rows) is given by the following matrix.")
print(NB.theta_)

print("The variance of each predictor (columns) for each class (rows) is given by the following matrix.")
print(NB.var_)

print("double check: ")
print(X_train[L_train=='Down'].mean())
print(X_train[L_train=='Down'].var(ddof=0))

nb_labels = NB.predict(X_test)
print(confusion_table(nb_labels, L_test))
print(NB.predict_proba(X_test)[:5])
