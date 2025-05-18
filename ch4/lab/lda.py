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
print(L_train)


lda = LDA(store_covariance = True)
#lda.fit(X_train, L_train)
X_train  = X_train.drop(columns=['intercept'])
X_test = X_test.drop(columns=['intercept'])
lda.fit(X_train, L_train)

print(lda.means_)
print(lda.priors_)
print(lda.scalings_)

lda_pred = lda.predict(X_test)

print(confusion_table(lda_pred, L_test))
lda_prob = lda.predict_proba(X_test)
print(
np.all(
    np.where(lda_prob[:,1] >= 0.5,'Up', 'Down')== lda_pred
))

