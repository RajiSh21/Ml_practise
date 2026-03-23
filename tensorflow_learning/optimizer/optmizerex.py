import tensorflow as tf

def model(bias, weights, features=borrower_features):
    product =  tf.matmul(features, weights)
    return tf.activations.sigmoid(product+bias)

def loss_function(bias, weights, targets = default, features = borrower_features):
    predictions = model(bias, weights)
    return tf.keras.losses.binary_crossentropy(targets, predictions)

opt = tf.keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.9)
opt.minimize(lambda: loss_function(bias, weights), var_list=[bias, weights])