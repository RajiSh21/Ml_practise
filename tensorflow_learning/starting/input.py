import numpy as np
import pandas as pd


housing = pd.read_csv("kc_housing.csv")

housing = np.array(housing) #converting to numpy array

price = np.array(housing['price'], np.float32)

waterfront = np.array(housing['waterfront'], np.bool)