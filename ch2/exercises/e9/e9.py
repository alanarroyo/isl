import numpy as np
import pandas as pd

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
def get_properties(column):
    properties = {}
    properties['type'] = quant_or_qual(column)
    properties['max'] = np.max(column)
    properties['min'] = np.min(column)
    if column.dtype == 'float64':
        properties['std'] = round(np.std(column),2)
    else:
        properties['std'] = None
    return properties

for col_name in Auto.columns:
    column = Auto[col_name]
    col_info[col_name] = get_properties(column)


print(col_info)
