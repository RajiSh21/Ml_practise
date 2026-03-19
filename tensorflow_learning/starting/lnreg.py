import numpy as np
import tensorflow as tf


#defining the target and featuress
pricee = np.array(housing['price'], np.float32)
size = np.array(housing['sqft_living'], np.float32)

#define the intercept and slope
intercept = tf.Variable(0.1, np.float32)
slope = tf.Variable(0.1, np.float32)

#defining the linear regression model
def linear_regression(intercept, slope, features = size):
    return intercept + slope * features


def loss_function(intercept, slope, targets = pricee, features = size):
    predictions = linear_regression(intercept, slope,)
    return tf.keras.losses.mse(targets, predictions)

opt = tf.keras.optimizers.Adam()

for j in range(1000):
    opt.minimize(lambda: loss_function(intercept, slope),\
                var_list=[intercept, slope])
    print(loss_function(intercept, slope))


print(intercept.numpy(), slope.numpy())