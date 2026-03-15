import tensorflow as tf
import pandas as pd
import numpy as np


intercept = tf.Variable(0.1, tf.float32)
slope = tf.Variable(0.1, tf.float32)

def linear_regression(intercept, slope, features):
    return intercept + features * slope

def loss_function(intercept, slope, features, targets):
    predictions = linear_regression(intercept, slope, features)
    return tf.keras.losses.mse(targets, predictions)

opt = tf.keras.optimizers.Adam(0.5)

for batch in pd.read_csv('kc_housing.csv', chunksize=100):
    price = np.array(batch['price'], np.float32)
    size = np.array(batch['size'], np.float32)

    
    opt.minimize(lambda: loss_function(intercept, slope, size, price), 
                 var_list=[intercept, slope])
    

print(intercept.numpy(), slope.numpy())