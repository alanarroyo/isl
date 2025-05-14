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

Boston = pd.read_csv('~/Documents/isl/data/Boston.csv')
#print(Boston.info())

def get_results_lr(df,y, columns):
    X = MS(columns).fit_transform(df)
    model = sm.OLS(df[y], X)
    results = model.fit()
    return results

significant = []
single_coeff = {'intercept':0}
for column in Boston.columns.drop('crim'):
    results =  get_results_lr(Boston, 'crim',[column])
    print(summarize(results ))
    if results.pvalues.iloc[1]<0.05:
        significant.append(column)
    single_coeff[column] = results.params[1]

print(single_coeff)
print('significant associated columns: ',  significant)

all_predict = Boston.columns.drop('crim')
results_all = get_results_lr(Boston, 'crim', all_predict)
print(summarize(results_all))

print(results_all.params)

coeff_df=pd.DataFrame(results_all.params, columns=['all']).reset_index()
coeff_df.rename(columns={'index':'predictor'}, inplace=True)

coeff_df['single']= coeff_df.predictor.map(lambda x: single_coeff[x])
print(coeff_df)

fig, ax = subplots(figsize=(8,8))
ax.scatter(coeff_df['single'], coeff_df['all'])
ax.set_xlabel('single predictor coefficient')
ax.set_ylabel('multiple predictor coefficient')
offset=5
for i in range(0, len(coeff_df)):
    ax.text(coeff_df.loc[i, 'single']-offset, coeff_df.loc[i, 'all'], coeff_df.loc[i,'predictor'])
    

fig.savefig('comparison_coeff.png')

for column in all_predict:
    results = get_results_lr(Boston,'crim', [poly(column, degree=2)] )
    print(summarize(results))
