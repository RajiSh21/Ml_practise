import tensorflow as tf


inputs = tf.constant(data, tf.float32)

dense1 = tf.keras.layers.Dense(10, activation='sigmoid')(inputs)

dense2 = tf.keras.layers.Dense(5, activation='sigmoid')(dense1)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)