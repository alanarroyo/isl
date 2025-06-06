import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt

Auto = pd.read_csv('~/Documents/isl/data/Auto.data', na_values = ['?'], delim_whitespace = True)
print(Auto)

#(a, b, c) quantitative vs qualitative
print(Auto.info())
col_info = {}

def quant_or_qual(column):
    if column.dtype == 'float64':
        return 'quantitative'
    elif column.dtype == 'int64':
        return 'qualitative'
    else:
        return 'other'
def trim(column):
    q1 = column.quantile(0.1)
    q2 = column.quantile(0.85)
    print(q1< column)
    return column[(q1 < column) & (column < q2)]

def trimmed_min(column):
    if column.dtype != 'float64':
        return None
    return np.min(trim(column))

def trimmed_max(column ):
    if column.dtype != 'float64':
        return None
    return np.max(trim(column))

def trimmed_std(column):
    if column.dtype != 'float64':
        return None
    else:
        return round(np.std(trim(column)), 2)

print( 'HEY  ', np.max(Auto.mpg),
     trimmed_max(Auto.mpg),
     trimmed_min(Auto.mpg),
     trimmed_std(Auto.mpg))
        
def get_properties(column):
    properties = {}
    properties['type'] = quant_or_qual(column)
    properties['max'] = np.max(column)
    properties['min'] = np.min(column)
    properties['tmax'] = trimmed_max(column)
    properties['tmin'] = trimmed_min(column)
    properties['tstd'] = trimmed_std(column)
    if column.dtype == 'float64':
        properties['std'] = round(np.std(column),2)
    else:
        properties['std'] = None
    return properties

for col_name in Auto.columns:
    column = Auto[col_name]
    col_info[col_name] = get_properties(column)


print(col_info)

# (c) investigate the predictors

#Let us first start by plotting a scatter_plot mattrix
#fig, ax = subplots() 
#pd.plotting.scatter_matrix(Auto, ax=ax, figsize=(100,100 ), alpha=0.7 )
#fig.savefig('scatter_matrix.png')
fig, ax = subplots()
ax = plt.scatter(Auto.horsepower, Auto.mpg, alpha=0.7,
           facecolors= 'none', edgecolors ='k' )
fig.savefig('scatterplot_mpg_vs_hoserpower.png', dpi=100)
"""
Observations:
    - displacement horsepower and weight seem to be positively correlated.
    - Previous 3 predictors seem to also be relared to the number of cylinders. 
    - These three variables are inversely related to mpg. This kind of make sense
. 
- It is lessclear if these tgre variables are inversely related to acceleration.
- There also seem to be some correlatin between mpg and year (the newer the more efficients are the cars?

"""
