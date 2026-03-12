import tensorflow as tf

x =tf.Variable(-1.0)

#defining y within instance of GradientTape
with tf.GradientTape() as tape:
    tape.watch(x)
    y=tf.multiply(x,x)

#evaluate the gradient of y at x=-1
g=tape.gradient(y,x)
print(g.numpy())


