import tensorflow as tf

#defininf a linear regression model
def linear_regression(interecept, slope = slope, features = features):
    return intercept + features*slope

#defining a loss function to calc MSE
def loss_function(intercept, slope, targets = targets, features = features):
    #compute predtciton for a linear model
    predictions = linear_regression(intercept, slope)

    return tf.keras.losses.mse(targets, predictions)

#compute the loss for teest data_ inputs
loss_function(interecept, sloper, test_targets, test_features)

#compute the loss for deafult inputs
loss_function(interecept, slope)
