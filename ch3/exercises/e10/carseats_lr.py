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

carseats = pd.read_csv('~/Documents/isl/data/Carseats.csv')
carseats.drop(columns=['Unnamed: 0'], inplace=True)
for column in ['Urban', 'US', 'ShelveLoc']:
    carseats[column] = carseats[column].astype('category')
print(carseats)

print(carseats.info())
# (a)

X = MS(['Price', 'Urban', 'US'
]).fit_transform(carseats)
y = carseats.Sales
print(X)
model_a = sm.OLS(y,X)
results_a = model_a.fit()
print(summarize(results_a))

#(d) 
predictors = carseats.columns.drop('Sales')
best_predictors = []
for column in predictors:
    X = MS([column]).fit_transform(carseats)
    model = sm.OLS(y,X)
    results = model.fit()
    print(summarize(results))
    pvalue = results.pvalues.iloc[1]
    if pvalue < 0.05:
        best_predictors.append(column)

print('Best predictors: ', best_predictors )
X =  MS(best_predictors).fit_transform(carseats)
model_e = sm.OLS(y,X)
results_e = model_e.fit()
print(summarize(results_e))

# (f) 
print('Sum of residuals for model a is: ',  (results_a.resid**2).sum())

print('Sum of residuals for model e is: ',  (results_e.resid**2).sum())
