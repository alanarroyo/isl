import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence \
	import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
			summarize,
			poly)
from sklearn.linear_model import LinearRegression

Boston =load_data('Boston')
X = MS(['lstat','age']).fit_transform(Boston)
y = Boston['medv']
print(X)

model1 = sm.OLS(y,X)
results1 = model1.fit()
print(summarize(results1 ))

terms = Boston.columns.drop('medv')
print(terms)

X = MS(terms).fit_transform(Boston)
model1 = sm.OLS(y,X)
results1 = model1.fit()
print(summarize(results1))

vals=[VIF(X,i) for i in range(1, X.shape[1])]
vif = pd.DataFrame({'vif':vals }, index=X.columns[1:] )
print(vif)

X = MS(['lstat', 'age', ('lstat','age')]).fit_transform(Boston)
model2 = sm.OLS(y,X)
print(summarize(model2.fit()))

X = MS([poly('lstat', degree=2), 'age']).fit_transform(Boston)
model3 = sm.OLS(y,X)
results3 = model3.fit()
print(summarize(results3))

print(anova_lm(results1,results3))

fig, ax = subplots(figsize=(8,8))
ax.scatter(results3.fittedvalues, results3.resid)
ax.set_xlabel('Fitted values')
ax.set_ylabel('Residual')
ax.axhline(y=0,c='k', ls='--')
fig.savefig('residuals_degree2.png')


