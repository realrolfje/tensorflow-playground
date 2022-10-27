#!python
print("Importing Tensorflow...")

from signal import signal
import tensorflow as tf
import numpy as np

import collections
import random

print("----------------------------------------")
print("TensorFlow version:", tf.__version__)
tf.constant(0) # get rid of the initial warning
print("----------------------------------------")

def isNaN(num):
    return num!= num

# Returns an array with 6 random floats between 0 and 1
def generate_random_input():
    v = [random.random() for i in range(6)]
    for n in v:
        if isNaN(n):
            raise("NaN in input.")
    return v

# Returns a predictable output for an array of 6 floats
def generate_predictable_output(i):
    v = [(i[0]+i[1])/2, (i[2]+i[3]), (i[4]+i[5])/2 ]
    for n in v:
        if isNaN(n):
            raise("NaN in output.")
    return v

# Generate 100 inputs (called x in Tensorflow)
inputs = [generate_random_input() for i in range(1000)]
inputshape = (len(inputs[0]),)


# Generate 100 predicted outcomes (called y in tensorflow)
outputs = [generate_predictable_output(e) for e in inputs]
outputunits = len(outputs[0])

# Create a sequential model (no idea if this is the correct one)
model = tf.keras.models.Sequential([
  tf.keras.layers.InputLayer(input_shape=inputshape), # Input layer has shape of inputs
#   tf.keras.layers.Dense(name="inner", units=outputunits*2, activation='relu'),
  tf.keras.layers.Dense(name="outputs", units=outputunits, activation='relu')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(inputs, outputs, epochs=100)

print('---------- model information -----------')
print(model.summary())

# [1,1,1,1,1,1] -> [1,2,1]
for i in range(4):
    x = generate_random_input()
    y = generate_predictable_output(x)
    print("   Predict inputs: ", x)
    print(" Expected outputs: ", y)
    print("Predicted outputs: ", model.predict([x]))
    print()
