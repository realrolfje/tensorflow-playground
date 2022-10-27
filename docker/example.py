#!python
print("Importing Tensorflow...")

from signal import signal
import tensorflow as tf

import collections
import random

print("----------------------------------------")
print("TensorFlow version:", tf.__version__)
tf.constant(0) # get rid of the initial warning
print("----------------------------------------")

# Stations, alphabetically sorted
station_names = sorted({"John", "Jack", "Peter", "X"})

# Reveivers to consider, ordered
receiver_names = sorted({"Groningen", "Leeuwarden", "Smilde", "Goes"})

# Turns the given receiver SNR values into a normalized Tensor.
# The Tensor contains only the SNRs for known receivers, ordered by receiver name
# as a one dimensional matrix of floating points between 0 and 1.
def signalToTensor(signal):
    filtered_signal = {k: v for k, v in signal.items() if k in receiver_names}
    filtered_snrs = collections.OrderedDict(sorted(filtered_signal.items()))
    if (len(filtered_snrs) != len(receiver_names)):
        raise("Not all receivers have a signal.")
    normalized = [v/100 for v in list(filtered_snrs.values())]
    tensor = tf.constant(normalized)
    # print(tensor)
    return normalized

# Tuns the given stationname into a normalized Tensor.
def stationToTensor(stationName):
    if stationName in station_names:        
        v = list(map(lambda v: 1.0 if (v == stationName) else 0.0, station_names)) + [0.0]
    else: 
        v = list(map(lambda v: 0.0, station_names)) + [1.0]    
    tensor = tf.constant(v)
    # print('Tensor for '+stationName, tensor)
    return v

# Splits the dataset into training, validation and test sets, 
# see https://towardsdatascience.com/how-to-split-a-tensorflow-dataset-into-train-validation-and-test-sets-526c8dd29438
def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

# Example data (this is what we need to build, with information from the svxlink snr values
# and manual classification of the audio.
# Proposed format is:
# date/time : stationname
# data/time : receiver snrs
# and then build a set like this:
inputdata = {
    "John": {"Groningen": 50, "Leeuwarden": 43, "Smilde": 29, "Goes": 12, "Ignored": 123},
    "Jack": {"Groningen": 48, "Leeuwarden": 50, "Smilde": 20, "Goes": 8, "Ignored": 123},
    "Peter": {"Groningen": 5, "Leeuwarden": 90, "Smilde": 1, "Goes": 3, "Ignored": 3},
}


def createTestData(size):
    def addnoise(s):
        noisepercentage = 0.1
        return s * (1 + noisepercentage * (1 - random.random() * 2) )

    stations = []
    signals = []

    for i in range(size):
        station = random.choice(list(inputdata.keys()))
        signal = inputdata[station]
        signal_noised = dict(map(lambda kv: (kv[0], addnoise(kv[1])), signal.items()))      
        stations.append(station)
        signals.append(signal_noised)
    
    return list(map(stationToTensor, stations)), list(map(signalToTensor, signals))



# stations = list(map(stationToTensor, inputdata.keys()))
# signals = list(map(signalToTensor, inputdata.values()))

stations, signals = createTestData(100)

print('-------- first row of station tensors -------------')
print(stations[0])
print(tf.constant(stations[0]))

print('-------- first row of signal tensors -------------')
print(signals[0])

print('---- first value from dataset ------')
inputs_dataset = tf.data.Dataset.from_tensor_slices(signals)
outputs_dataset = tf.data.Dataset.from_tensor_slices(stations)
dataset = tf.data.Dataset.zip((inputs_dataset, outputs_dataset))

# dataset = tf.data.Dataset.from_tensor_slices((signals, stations))

# dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4], [1,2,3,4,5])

print('-------- dataset from signals and stations tensors -------------')
print(dataset.take(1))

# exit(1)

# Split and shuffle the dataset into training, validation and testing sets
# dataset_training, dataset_validation, dataset_test = get_dataset_partitions_tf(dataset, len(signals))

# print('-------- training dataset -------------')
# print(dataset_training.take(1))

# print('-------- validation dataset -------------')
# print(dataset_validation.take(1))

# print('-------- test dataset -------------')
# print(dataset_test.take(1))

# Create a sequential model (no idea if this is the correct one)
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(name="inputs", units=len(receiver_names), activation='relu'),
#   tf.keras.layers.Dense(units=len(station_names)*2, activation='relu'),
  tf.keras.layers.Dense(name="outputs", units=len(station_names)+1, activation='softmax')
])

print("------------- Compiling")
model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
)

# model.fit(dataset_training, epochs=5)
print("Number of signal rows  : %s" % len(signals))
print("Number of station rows : %s" % len(stations))
print("--------------- model.fit")
# model.fit([tf.constant(v) for v in signals], [tf.constant(v) for v in stations], epochs=5)

x = tf.stack([tf.constant(v) for v in signals])
y = tf.stack([tf.constant(v) for v in stations])

x = [tf.constant(signals[0])]
y = [tf.constant(stations[0])]

model.fit(x, y, epochs=5)

print('---------- model information -----------')
print(model.summary())


print("---- Done training, lets test: -----")
print(signals[1])
print(stations[1])
print(model.predict(signals[1]))


# print("Jack transmits:")
# print(model.predict(signalToTensor(inputdata["Jack"])))

# print("John transmits:")
# print(model.predict(signalToTensor(inputdata["John"])))

# print("Peter transmits:")
# print(model.predict(signalToTensor(inputdata["Peter"])))

# print("Some other station:")
# print(model.predict(signalToTensor({"Groningen": 1, "Leeuwarden": 1, "Smilde": 1, "Goes": 3, "Ignored": 3})))

# does not work yet: model.evaluate(dataset_test)


