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

Auto = pd.read_csv("~/Documents/isl/data/Auto.data" ,sep='\s+',  # delim_whitespace=True,
na_values=['?'])
Auto = Auto.dropna(how='any')
print(Auto)

X = MS(['horsepower']).fit_transform(Auto)
print(X)
y = Auto.mpg
model = sm.OLS(y,X)
results = model.fit()

print(Auto[['mpg', 'horsepower']].describe())
print(summarize(results))

test = pd.DataFrame({'horsepower':[98]})
newX = MS(['horsepower']).fit_transform(test)
pred = results.get_prediction(newX)
print('prediction for horsepower=98 is equal to mpg=', pred.predicted_mean[0])

ci = pred.conf_int(alpha=0.05)
print('with a 95%  confidence interval: ', ci[0])


# (b) plot response and the predictor

fig, ax = subplots(figsize = (8,8))
ax.scatter(X['horsepower'], y,   
        facecolors='none', edgecolors='k'
)
print(results.params)
ax.axline([0,results.params.iloc[0]],
slope = results.params.iloc[1] , color='r')
ax.set_xlabel('Horsepower')
ax.set_ylabel('mpg (miles per gallon)')
fig.savefig('hp_vs_mpg.png',
        )


# (c) diagnostic plot

fig1, ax1 =  subplots(figsize = (8,8))
ax1.scatter(results.fittedvalues, results.resid)
ax1.set_xlabel('Fitted value')
ax1.set_ylabel('Residuals')
ax1.axline((0,0),slope=0, c='k', ls='--' )
fig1.savefig('residual.png')

infl = results.get_influence()
#print(infl.hat_matrix_diag)
print(np.arange(X.shape[0]))
fig2, ax2 = subplots(figsize = (8,8))
ax2.scatter = (np.arange(X.shape[0]),infl.hat_matrix_diag )
ax2.set_xlabel('Index')
ax2.set_ylabel('Influence')
fig2.savefig('influence.png')
