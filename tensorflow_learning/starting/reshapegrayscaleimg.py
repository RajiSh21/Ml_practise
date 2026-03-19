import tensorflow as tf


#generte grayscale image
gray = tf.random.uniform([2,2], maxval=255, dtype='int32')

#reshape grayscale image
gray =tf.reshape(gray, [2*2, 1])