#!python
import tensorflow as tf
import tensorflow_datasets as tfds

print("TensorFlow version:", tf.__version__)

# //mnist = tf.keras.datasets.mnist
mnist = tfds.load('mnist', split='train', shuffle_files=True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0