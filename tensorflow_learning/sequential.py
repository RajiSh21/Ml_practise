#Building a sequential model
from turtle import mode

from tensorflow import keras

#sequential model
model = keras.Sequential()

#first hidden layer
model.add(keras.layers.Dense(16, activation='relu', input_shape=(28*28,)))

#second hidden layer
model.add(keras.layers.Dense(8, acctivation='relu'))

#define output layer
model.add(keras.layers.Dense(4, activation='spftmax'))

#compile the model
model.compile('adam', loss='categorical_crossentropy')

#summarize the model;
print(model.summary())