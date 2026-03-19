import tensorflow as tf

inputs = tf.constant([[1.0, 2.0]])

weights = tf.Variable([[-0.5], [-0.01]])

bias = tf.Variable([0.5])

product = tf.matmul(inputs, weights)

dense = tf.keras.activations.sigmoid(product + bias) #dense layer
