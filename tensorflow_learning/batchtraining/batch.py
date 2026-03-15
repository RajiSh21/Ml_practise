import pandas as pd
import numpy as np

for batch in pd.read_csv('kc_housing.csv', chunksize=100):
    price = np.array(batch['price'], np.float32)

    size = np.array(batch['size'], np.float32)