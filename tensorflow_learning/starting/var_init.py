#initializing the variables in tensorflow

import tensorflow as tf
weights = tf.Variable(tf.random.normal([500,500]))

weights = tf.Variable(tf.random.truncated_normal([500,500]))
#defining a dense layer with the default initialize
dense = tf.keras.layers.Dense(32, activation='relu')

#dense layer with zeros initializer
dense = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='zeros')