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

Carseats = load_data('Carseats')
print(Carseats.columns)

allvars = list(Carseats.columns.drop('Sales'))
y = Carseats.Sales
final = allvars + [('Income', 'Advertising'),
                    ('Price', 'Age')]
X = MS(final).fit_transform(Carseats)
model = sm.OLS(y,X)
print(summarize(model.fit()))
