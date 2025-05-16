
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

allvars = Smarket.columns.drop(['Today', 'Direction', 'Year'])
design = MS(allvars)
X = design.fit_transform(Smarket)
y = Smarket.Direction == 'Up'
glm = sm.GLM(y,X, family=sm.families.Binomial() )
results = glm.fit()
print(summarize(results))
print(results.params)
print(results.pvalues)
probs = results.predict()
print(probs[:10])

def convert_to_up_down(x):
    if x>0.5:
        return 'Up'
    else:
        return 'Down'

convert_to_up_down_v = np.vectorize(convert_to_up_down)

labels = convert_to_up_down_v(probs)
print(labels[:10])

print(confusion_table(labels, Smarket.Direction))
print( np.mean(labels == Smarket.Direction))

# train test first split
train  = (Smarket.Year<2005)
Smarket_train = Smarket.loc[train]
Smarket_test = Smarket.loc[~train]

X_train, X_test = X.loc[train], X.loc[~train]
y_train, y_test = y.loc[train], y.loc[~train]
glm_train = sm.GLM(y_train, X_train, 
            family = sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog = X_test)

D = Smarket.Direction
L_train, L_test = D.loc[train], D.loc[~train]

labels = convert_to_up_down_v(probs)
print(confusion_table(labels, L_test))
print(
np.mean(labels ==L_test), 
np.mean(labels != L_test)
)

