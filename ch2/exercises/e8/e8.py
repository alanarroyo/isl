
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt

# (a) read data

college = pd.read_csv('~/Documents/isl/data/College.csv',
                     # index_col=0
                     )
print('Our dataframe is the following:')
print(college)

# (b) fix first column name

college = college.rename({'Unnamed: 0': 'College'}, axis=1)
college = college.set_index('College')
print(college)

# (c) describe

print( college.describe())

# (d) scatter plot matrix

#fig, axes = subplots(figsize = (8,8))
pd.plotting.scatter_matrix(college.loc[:, ['Top10perc', 'Apps', 'Enroll']])
plt.savefig('scatter_matrix.png')

# (e) boxplot of outstate vs private
college.Private = pd.Series(college.Private, dtype ='category')
print(college.info())
fig, ax = subplots(figsize= (8,8))
college.boxplot('Outstate',by= 'Private',ax=ax )
fig.savefig('boxplot.png')
