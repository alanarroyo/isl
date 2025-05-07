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
from itertools import combinations

Auto = pd.read_csv("~/Documents/isl/data/Auto.data" ,sep='\s+',  # delim_whitespace=True,
na_values=['?'])
Auto = Auto.dropna(how='any')
print(Auto)

fig, ax = subplots(figsize=(20,20))

pd.plotting.scatter_matrix( Auto, ax=ax)

fig.savefig('scatter_matrix.pdf',format='pdf' )

# (b) 
numeric_col = Auto.columns.drop('name')
Auto_num = Auto[numeric_col]
print(Auto_num.corr())

# (c)

X = MS(Auto_num.columns.drop('mpg')).fit_transform(Auto_num)
y = Auto['mpg']
print(X)
model = sm.OLS(y,X)
results = model.fit()
print(summarize(results))
print('F statistic: ', results.fvalue)

# (d) diagnostic plots

fig1, ax1 = subplots(figsize=(8,8))
ax1.scatter(results.fittedvalues, results.resid )
ax1.axline((0,0), slope=0, color = 'k', ls='--')
ax1.set_xlabel('Fitted value (mpg)')
ax1.set_ylabel('Residual')
fig1.savefig('residuals.png')

fig2, ax2 = subplots(figsize=(8,8))
infl = results.get_influence()
ax2.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag )
ax2.set_xlabel('Index')
ax2.set_ylabel('Leverage')
fig2.savefig('leverage.png')
print(np.argmax(infl.hat_matrix_diag))

# (e) )

deg1_cols = list(Auto_num.columns.drop('mpg'))
interaction = list( combinations(deg1_cols, 2) )

X = MS(deg1_cols+interaction).fit_transform(Auto_num)
model = sm.OLS(y,X)
results = model.fit()
print(summarize(results))
