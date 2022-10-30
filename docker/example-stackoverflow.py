#!python

#
# For Stackoverflow question:
# https://stackoverflow.com/questions/74251555/how-to-correctly-train-and-predict-in-tensorflow-2
#

print("Importing Tensorflow...")
import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)
tf.constant(0) # get rid of the initial warning

# The names I want to identify
names = sorted({"Jack", "John", "Peter", "X"})

# The sizes for each of the names
sizes = [
            [0.0, 0.5, 0.6, 0.7, 0.8, 0.3], # Jack's sizes
            [0.2, 0.6, 0.7, 0.8, 0.5, 0.2], # John's sizes
            [0.0, 0.1, 0.1, 0.4, 0.8, 0.9], # Peter's sizes
            [0.3, 0.9, 0.2, 0.1, 0.0, 0.8]  # X's sizes
]

# As far as I understand, the input layer is the number of inputs per case (sizes in my case)
# The outputs is the number of names, where each output represents the "probability" between 0 and 1
# that the inputs match this particular name (so unit 0 represents Jack's probability, etc etc)
# 
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(name="inputs",  units=len(sizes[0]), activation='relu'),
  tf.keras.layers.Dense(name="outputs", units=len(names), activation='relu')
])

model.compile(
        optimizer='adam',
        # Uses one-hot, see https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy
        loss=tf.keras.losses.CategoricalCrossentropy(), 
        metrics=['accuracy']
)

# This trainingset is way too small, but even with this set, tensorflow gives up
x = np.array(sizes) # Stack size Tensors for all names
y = np.array([ # Stack categorization probabilities for each x -> y
        [1.0, 0.0, 0.0, 0.0], # Jack's index
        [0.0, 1.0, 0.0, 0.0], # John's index
        [0.0, 0.0, 1.0, 0.0], # Peter's index
        [0.0, 0.0, 0.0, 1.0]  # X's index
    ])

# This works with one-hot CategoricalCrossentropy loss function
model.fit(x, y, epochs=100)
print('---------- model information -----------')
print(model.summary())

print("---- Done training, lets test: -----")
xtest = [sizes[0]] # Jack's sizes as a Tensor matching the 6 inputs
print('   input: ',xtest)
ytest = model.predict(xtest)  # Expect this to be [1.0, 0.0, 0.0, 0.0], Jack.
print('  actual: ', ytest)
print('expected: ', [y[0]])


