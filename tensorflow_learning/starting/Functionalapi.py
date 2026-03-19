#Using the functional APi
import tensorflow as tf

#model 1 input layer shpae
model1_inputs =  tf.keras.Input(shape=(28*28))

#define model 2 input layers shape
model2_input = tf.keras.Input(shpae=((10,))

#define layer 1 for model 1
model1_layer1 = tf.keras.layers.Dense(12, activation='relu')(model1_inputs)

#define layer2 for model 2
model1_layer2 = tf.keras.layers.Dense(4, activation='softmax')(model1_layer1)

#layer1 for model2
model2_layer1 = tf.keras.layers.Dense(8, activation='relu')(model2_input)

#layer2 for model2
model2_layer2 = tf.keras.layes.Dense(4, activation='softmax')(model2_layer1)

#merging model 1 and model 2
merged = tf.keras.layers.add([model1_layer2, model2_layer2])

#define a functional model
model=tf.keras.Model(inputs=[model1_inputs,model2_inputs], output = merged)

#compile the model
model.compile('adam' , loss='categorical_crossentropy')