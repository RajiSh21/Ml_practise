import numpy as np
import tensorflow as tf

yound, old = 0.3, 0.6
low_bill, high_bill = 0.1, 0.5

young_high =1.0*young + 2.0*high_bill
young_low = 1.0*young = 2.0*low_bill
old_high = 1.0*old + 2.0*high_bill
old_low = 1.0*old + 2.0*low_bill

print(young_high - young_low)


print(old_high - old_low)

#difference in deafult predictions for young
print(tf.keras.activations.sigmoid(young_high ).numpy() -tf.keras.activations.sigmoid(young_low).numpy())

#difference in deafult predictions for old
print(tf.keras.activations.sigmoid(old_high).numpy() - tf.keras.activations.sigmoid(old_low))

