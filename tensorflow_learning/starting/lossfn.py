import tensorflow as tf

#calc the MSE loss
loss = tf.keras.losses.MeanSquaredError(targets, predictions)

