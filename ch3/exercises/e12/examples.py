import numpy as np
import pandas as pd
import  statsmodels.api as sm 

from ISLP.models import (ModelSpec as MS,
			summarize,
			poly)
x = np.random.normal(size = 100)
y1 = 2*x
y2 = -x

model1 = sm.OLS(y1,x)
model2 = sm.OLS(y2,x)
model1i = sm.OLS(x,y1)
model2i = sm.OLS(x,y2)

result1 = model1.fit()
result2 = model2.fit()
result1i = model1i.fit()
result2i = model2i.fit()

print(summarize(result1))
print(summarize(result1i))

print(summarize(result2))
print(summarize(result2i))
