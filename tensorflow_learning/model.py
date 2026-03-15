# Define a linear regression model
def linear_regression(intercept, slope, features):
    return intercept + features * slope

# Set loss_function() to take the variables as arguments
def loss_function(intercept, slope, features, targets):
    predictions = linear_regression(intercept, slope, features)
    return keras.losses.mse(targets, predictions)

# Initialize an Adam optimizer
opt = keras.optimizers.Adam(0.5)

for j in range(100):
    opt.minimize(lambda: loss_function(intercept, slope, size_log, price_log), var_list=[intercept, slope])
    if j % 10 == 0:
        print(loss_function(intercept, slope, size_log, price_log).numpy())