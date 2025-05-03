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
pred = results.predict(newX)
print('prediction for horsepower=98 is equal to mpg=', pred) 
