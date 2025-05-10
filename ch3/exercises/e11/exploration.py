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

rng = np.random.default_rng(1)
x = rng.normal(size = 100)
y = 2* x+ rng.normal(size=100)

model1 = sm.OLS(y,x)
results1 = model1.fit()
print(summarize(results1))
print('p value of coefficient: ', results1.pvalues[0])

# (b) 
model2 = sm.OLS(x,y)
results2 = model2.fit()
print(summarize(results2))

print('p value of coefficient: ', results2.pvalues[0])
