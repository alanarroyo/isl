import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt

# (a) load Boston dataset

boston = pd.read_csv('~/Documents/isl/data/Boston.csv')
print(boston)

#(b) number of rows, columns and what do they represent?

print('Number of rows: ', len(boston))
print('Number of columns: ', len(boston.columns))
explanation = """
Rows represent a single suburb in Boston.
Each column are characterisitcs of each suburbm focused on housing values and aspects that can alter their prices.
"""
print(explanation)

#(c) Scatter plots
boston['log_crim']=np.log(boston.crim)
fig1, ax1 = subplots(figsize=(50,50))
#pd.plotting.scatter_matrix(boston, ax = ax1)
#fig1.savefig('scatter_matrix.pdf', format='pdf') 

#(d) and (e) Findings

findings = """
(1) The more residencial zoning the less crimes
(2) The more percentage of industry the more crimes, except when percentage is close to maximumn
(3) Being close to river might not be a good predictor, onless crime is high.
(4) Nitrogen oxide seems to be linearly correlated to the log of crime index.
(5) Average room size might be not a good predictor. however, low rome size is associated to high crims and high room size with low crimes
(6) The more percentage of old buildings the more crime
(7) The less distance to job centers the more crimes
(8) The farest from highways the more crimes, it seems, howver, is not very clear.
(9) The higher tax per 100,000, the more crimes, it seems
(10) unclear relation with ptratio
(11) lower status is positively correlated with crime rates
(12) median value of homes is seem to be inversely related to crimes, however, the relationship is not linear`
"""

print(findings)

#(e)
boston_sub = boston[['crim','ptratio','tax']]
print(boston[['crim', 'ptratio','tax']].describe())

print('crime 90% percentile', np.percentile(boston.crim,90))
print(boston_sub.sort_values(by='crim', ascending=False )[:20])  

# (f) suburvs bound the Charles river

number_river = (boston.chas == 1).sum()
print('Suvurbs next to river: ', number_river )

# (g) median pupil-teacher ration
median_ptratio = np.median(boston.ptratio)

print('Median pupil-teacher ratio: ', median_ptratio)

# (h) special suburb
special = boston[boston.medv == np.min(boston.medv)]
print(special)
special_compare = pd.concat([special, boston.describe()])
print(special_compare)
i_predictor=[]
for column in special_compare.columns:
	special_value = special_compare.loc[398,column]
	#print(special_value)
	if special_value < special_compare.loc['25%',column] or special_value > special_compare.loc['75%',column]:
		print(special_compare[[column]])
