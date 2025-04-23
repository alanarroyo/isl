import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import subplots
fig, ax = subplots(figsize=(8,8))
rng = np.random.default_rng(3)
x = rng.standard_normal(100)
y = rng.standard_normal(100)
ax.plot(x,y)
fig.savefig('first_graphics.png')

#scatter plot
ax.plot(x,y,'o')
fig.savefig('scatterplot.png')

# fig with labelled axis
ax.set_xlabel("this is the x-axis")
ax.set_ylabel("this is the y-axis")
ax.set_title("Plot of X vs Y")

fig.set_size_inches(12,3)
fig.savefig('scatterplot_with_axis.png')

# several charts in same figure
fig, axes  = subplots(nrows = 2,
                      ncols = 3,
                      figsize = (15,5))
axes[0,1].plot(x,y,'o')
axes[1,2].plot(x,y, marker = '+')

axes[0,1].set_xlim([-1,1])

fig.savefig('many_plots.png')

# 3D plot
fig, ax = subplots(figsize = (8, 8) )
x = np.linspace ( -np.pi, np.pi, 50 )
y = x
f = np.multiply.outer(np.cos(y), 1 / (1+x**2))
ax.contour (x, y,  f )

fig.savefig('3D_plot.png')
