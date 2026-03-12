import numpy as np 
import tensorflow as tf

#input data 
inputs = np.array(borrower_feature, np.float32)

#dense layer 1
dense1 =tf.keras.layers.Dense(32, activation='relu') (inputs)

#dense layer 2
dense2 = tf.keras.layers.Dense(12, activation='relu')(dense1)

#applying dropout operation
dropout1 = tf.keras.layers.Dropout(0.25)(dense2)

#op layer
outputs= tf.keras.layers.Dense(1, activation='sigmoid')(dropout1)