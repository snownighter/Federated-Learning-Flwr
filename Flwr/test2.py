import numpy as np
import pandas as pd

np.random.seed(42)
dir = np.random.dirichlet([30]*3, 25)
pd.DataFrame(dir).to_csv('test.csv')
