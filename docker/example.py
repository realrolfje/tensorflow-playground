#!python
print("Importing Tensorflow...")

import collections
import tensorflow as tf

print("----------------------------------------")
print("TensorFlow version:", tf.__version__)
tf.constant(0) # get rid of the initial warning
print("----------------------------------------")

# Stations, alphabetically sorted
stations = sorted({"John", "Jack", "Peter"})

# Reveivers to consider, ordered
receivers = sorted({"Groningen", "Leeuwarden", "Smilde", "Goes"})

# Turns the given receiver SNR values into a normalized Tensor.
# The Tensor contains only the SNRs for known receivers, ordered by receiver name
# as a one dimensional matrix of floating points between 0 and 1.
def signalToTensor(signal):
    filtered_signal = {k: v for k, v in signal.items() if k in receivers}
    filtered_snrs = collections.OrderedDict(sorted(filtered_signal.items()))
    if (len(filtered_snrs) != len(receivers)):
        raise("Not all receivers have a signal.")
    tensor = tf.constant(list(filtered_snrs.values()))/100
    # print(tensor)
    return tensor

# Tuns the given stationname into a normalized Tensor.
# The tensor contains the index of the name in the sorted list of stations,
# or an index just ourtside of that if it is not in the list.
def stationToTensor(stationName):
    try:
        index = stations.index(stationName)
    except:
        index = len(stations)
    tensor = tf.constant(index)
    # print(tensor)
    return tensor

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
    "Peter": {"Groningen": 5, "Leeuwarden": 90, "Smilde": 1, "Goes": 3, "Ignored": 123},
    "John": {"Groningen": 50, "Leeuwarden": 43, "Smilde": 29, "Goes": 12, "Ignored": 123},
    "Peter": {"Groningen": 5, "Leeuwarden": 90, "Smilde": 1, "Goes": 3, "Ignored": 123},
    "John": {"Groningen": 50, "Leeuwarden": 43, "Smilde": 29, "Goes": 12, "Ignored": 123},
    "Peter": {"Groningen": 5, "Leeuwarden": 90, "Smilde": 1, "Goes": 3, "Ignored": 123},
    "John": {"Groningen": 50, "Leeuwarden": 43, "Smilde": 29, "Goes": 12, "Ignored": 123},
    "Peter": {"Groningen": 5, "Leeuwarden": 90, "Smilde": 1, "Goes": 3, "Ignored": 123},
    "John": {"Groningen": 50, "Leeuwarden": 43, "Smilde": 29, "Goes": 12, "Ignored": 123},
    "Peter": {"Groningen": 5, "Leeuwarden": 90, "Smilde": 1, "Goes": 3, "Ignored": 123},
    "John": {"Groningen": 50, "Leeuwarden": 43, "Smilde": 29, "Goes": 12, "Ignored": 123},
    "Peter": {"Groningen": 5, "Leeuwarden": 90, "Smilde": 1, "Goes": 3, "Ignored": 123},
    "John": {"Groningen": 50, "Leeuwarden": 43, "Smilde": 29, "Goes": 12, "Ignored": 123},
    "Peter": {"Groningen": 5, "Leeuwarden": 90, "Smilde": 1, "Goes": 3, "Ignored": 123},
    "John": {"Groningen": 50, "Leeuwarden": 43, "Smilde": 29, "Goes": 12, "Ignored": 123},
    "Peter": {"Groningen": 5, "Leeuwarden": 90, "Smilde": 1, "Goes": 3, "Ignored": 123},
    "John": {"Groningen": 50, "Leeuwarden": 43, "Smilde": 29, "Goes": 12, "Ignored": 123},
    "Peter": {"Groningen": 5, "Leeuwarden": 90, "Smilde": 1, "Goes": 3, "Ignored": 123},
    "John": {"Groningen": 50, "Leeuwarden": 43, "Smilde": 29, "Goes": 12, "Ignored": 123},
    "Peter": {"Groningen": 5, "Leeuwarden": 90, "Smilde": 1, "Goes": 3, "Ignored": 123},
}

stations = list(map(stationToTensor, inputdata.keys()))
signals = list(map(signalToTensor, inputdata.values()))


print('-------- station tensors -------------')
print(stations)

dataset = tf.data.Dataset.from_tensors((signals, stations))

print('-------- dataset from signals and stations tensors -------------')
print(dataset)


# Split and shuffle the dataset into training, validation and testing sets
dataset_training, dataset_validation, dataset_test = get_dataset_partitions_tf(dataset, len(signals))

print('-------- training dataset -------------')
print(dataset_training)

print('-------- validation dataset -------------')
print(dataset_validation)

print('-------- test dataset -------------')
print(dataset_test)

# Create a sequential model (no idea if this is the correct one)
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(len(receivers)),
  tf.keras.layers.Dense(len(stations)*2, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(len(stations))
])

print(dataset_training)
print('---')


a = dataset_training.take(1)
print(a)
predictions = model.predict(a)
print(predictions)

