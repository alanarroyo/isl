
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

knn1 = KNeighborsClassifier(n_neighbors =1)
knn1.fit(X_train, L_train)
knn1_pred = knn1.predict(X_test)
print(confusion_table(knn1_pred, L_test))
print(np.mean(knn1_pred == L_test))

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3_pred = knn3.fit(X_train, L_train).predict(X_test)
print(np.mean(knn3_pred == L_test))

# caravan

Caravan =  load_data('Caravan')
Purchase = Caravan.Purchase
print(Purchase.value_counts())

feature_df = Caravan.drop(columns=['Purchase']
)
scaler = StandardScaler(with_mean=True,
                        with_std=True,
                        copy=True)
scaler.fit(feature_df)
X_std = scaler.transform(feature_df)
feature_std = pd.DataFrame(
                X_std,
                columns = feature_df.columns)
print(feature_std.std())

(X_train, 
X_test,
y_train, 
y_test) = train_test_split(feature_std, Purchase, test_size=1000, random_state=0)

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1_pred = knn1.fit(X_train, y_train).predict(X_test)
print(np.mean(y_test != knn1_pred))
print(np.mean(y_test != 'No'))
print(confusion_table(knn1_pred, y_test))
