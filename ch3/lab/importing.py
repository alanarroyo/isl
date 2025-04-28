import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import matplotlib
import statsmodels.api as sm
from statsmodels.stats.outliers_influence \
	import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
			summarize,
			poly)
print(dir())
A = np.array([3,5,11])
print(dir(A))
print(A.sum())
#print(A.sum?)
