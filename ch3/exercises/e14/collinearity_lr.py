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

rng = np.random.default_rng(10)
x1 = rng.uniform(0,1, size=100)
x2 = 0.5 * x1 + rng.normal(size=100)/10
y = 2 + 2 * x1 + 0.3 * x2 + rng.normal(size=100)

add_outlier = True
if add_outlier:
    x1 = np.concatenate([x1,[0.1]])
    x2 = np.concatenate([x2,[0.8]])
    y = np.concatenate([y, [6]])

fig, ax = subplots(figsize=(8,8))
ax.scatter(x=x1, y=x2)
ax.axline((0,0),slope=0.5, color='k' )
ax.set_xlabel('x1')
ax.set_ylabel('y1')
fig.savefig('x1_x2_scatter.png')

df = pd.DataFrame({'x1':x1, 'x2':x2})
X = MS(['x1', 'x2']).fit_transform(df)
model = sm.OLS(y, X)
results = model.fit()
print(summarize(results))
print('can reject null for beta1 not not for beta2')
# (d) and (e)

X1 = MS(['x1']).fit_transform(df)
X2 = MS(['x2']).fit_transform(df)
model1 = sm.OLS(y,X1)
model2 = sm.OLS(y,X2)
results1 = model1.fit()
results2 = model2.fit()

print(summarize(results1))
print(summarize(results2))
