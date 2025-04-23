
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

# (f) bining variable

college['Elite'] = pd.cut(college['Top10perc'],
                         [0,50,100],
                         labels = ['No', 'Yes'])
print(college['Elite'].value_counts())
fig, ax = subplots(figsize = (8,8))
college.boxplot('Outstate', by = 'Elite', ax=ax)
fig.savefig('boxplot_elite.png')

# (g) multiple plot

fig, axes = subplots(nrows = 2, 
                     ncols =2, 
                     figsize =(16,8) )
axes[0,0].hist(college['Outstate'], label='Outstate')
axes[0,0].set_title('Outstate')
axes[0,1].hist(college['PhD'])
axes[1,0].hist(college['Apps'])
axes[1,1].hist(college['Accept'])

fig.savefig('multiple_hist.png')
