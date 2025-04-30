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

Boston = load_data("Boston")
print(Boston.columns)

X = pd.DataFrame({'intercept':np.ones(len(Boston)),
'lstat':Boston['lstat']
})
print(X.head())

y = Boston['medv']
model = sm.OLS(y,X)
results = model.fit()
print(summarize(results))

design =  MS(['lstat'])
design = design.fit(Boston)
X = design.transform(Boston)

print(X.head())

#print(results.summary())
print(results.params)

# predictions

new_df = pd.DataFrame({'lstat':[5,10,15]})
newX = design.transform(new_df)
print(newX)
new_predictions = results.get_prediction(newX)
print(new_predictions.predicted_mean)
print( new_predictions.conf_int(alpha = 0.05))

reg = LinearRegression(fit_intercept=False)
reg.fit(X, y)

print('intercept:', reg.intercept_)
print('coefficients: ', reg.coef_)
print('predictions: ', reg.predict(newX))

# plot line

def abline(ax, b, m, *args, **kwargs):
    xlim = ax.get_xlim()
    ylim = [m + xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)
   
fig, ax = subplots(figsize=(8,8))
ax.scatter(Boston.lstat, Boston.medv,
            facecolors = 'none',
            edgecolors = 'k')
abline(ax, results.params.values[0], results.params.values[1],

    linewidth=3,
    color = 'r')
ax.set_xlabel('lstat: % lower status of the population ')
ax.set_ylabel("medv: Median value of home (in $1000's)")
ax.set_title('A simple regression')
fig.savefig('simple_reg.png')
