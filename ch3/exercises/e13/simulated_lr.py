import pandas as pd
import numpy as np

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
rng = np.random.default_rng(seed=1)
x = rng.normal(size=100)

eps = rng.normal(loc=0,scale =0.25, size=100)
eps_low = rng.normal(loc=0,scale =0.01, size=100)
eps_high = rng.normal(loc=0,scale =2, size=100)
y = -1 +0.5 *x +eps
y_low = -1 +0.5 *x +eps_low
y_high = -1 +0.5 *x +eps_high
print('beta_0 =-1, beta_1= 0.5')

fig, ax = subplots(figsize=(8,8))

ax.scatter(x=x, y=y)

#fig.savefig('scatter_plot.png')

df = pd.DataFrame({'x':x})
X = MS(['x']).fit_transform(df)
model = sm.OLS(y,X)
results = model.fit()
print(summarize(results))
print('beta_0_hat = -1.01, beta_1_hat= 0.49')

#(f)

ax.axline((0,-1), slope=0.5, color = 'k', label = 'real model')
ax.axline((0,results.params.iloc[0] ), slope = results.params.iloc[1], color = 'r', label = 'predicted model')
ax.legend()
fig.savefig('linear_regression.png')

X2 = MS([poly('x', degree=2)]).fit_transform(df)
model2 = sm.OLS(y, X2)
results2 = model2.fit()

print(summarize(results2))
print('p value of X^2 is high, showing that, undert the precense of X, X^2 does not influence the predicted Y')

model_low = sm.OLS(y_low, X)
results_low = model_low.fit()

model_high = sm.OLS(y_high, X)
results_high = model_high.fit()

print(summarize(results_low))
print(summarize(results_high))

print('CI normal: ', results.conf_int())
print('CI less noise: ', results_low.conf_int())
print('CI more noise: ', results_high.conf_int())
