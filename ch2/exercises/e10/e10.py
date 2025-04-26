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
pd.plotting.scatter_matrix(boston, ax = ax1)
fig1.savefig('scatter_matrix.pdf', format='pdf') 

# (
