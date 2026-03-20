import tensorlfow as tf 

inputs = tf.constant(borrower_feature, tf.float32)

dense1 = tf.keras.layers.Dense(16, activation='relu')(inputs)

dense2 = tf.keras.layers.Dense(8, activation='sigmoid')(dense1)

outputs = tf.keras.layers.Dense(4, activation='softmax')(dense2)