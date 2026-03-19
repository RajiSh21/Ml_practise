#defining feature columns
import tensorflow as tf

#numeric feature column
size=tf.feature_column.numeric_column("size")

#categorical feature columns
room = tf.feature_column.categorical_column_with_vocabulary_lsit("rows", ["1", "2", "3", "4", "5"])

#feature column list
feature_list = [size, room]

#matrix featire column
feature_list = [tf.feature_column.numeric_column('image', shape=(784,))]

#loading and transforming data
def input_fn():
    feature = {"size": [1340, 1690, 2720], "rooms":[1 ,3,4]}
    labels=[221900, 53800, 1800000]
    return feature_list, labels

#define a deep neual network regression
model0 = tf.estimator.DNNRegressor(feature_columns=feature_list, hidden_units = [10,6,6,3])

#train the regression model
model0.train(input_fn, steps=20)

#define a deep neural network classifier
model1 = tf.estimator.DNNClassifier(feature_columns=feature_list, hidden_units=[32, 16, 8], n_classes=4)
