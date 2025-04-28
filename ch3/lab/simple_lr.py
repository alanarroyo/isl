import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib
import statsmodels.api as sm
from statsmodels.stats.outliers_influence \
	import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
			summarize,
			poly)

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

print(results.summary())
print(results.params)

# predictions

new_df = pd.DataFrame({'lstat':[5,10,15]})
newX = design.transform(new_df)
print(newX)
new_predictions = results.get_prediction(newX)
print(new_predictions.predicted_mean)
print( new_predictions.conf_int(alpha = 0.05))
