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
    return tf.constant(list(filtered_snrs.values()))/100

# Tuns the given stationname into a normalized Tensor.
# The tensor contains the index of the name in the sorted list of stations,
# or an index just ourtside of that if it is not in the list.
def stationToTensor(stationName):
    try:
        index = stations.index(stationName)
    except:
        index = len(stations)
    return tf.constant(index)

# Example data (this is what we need to build, with information from the svxlink snr values
# and manual classification of the audio.
# Proposed format is:
# date/time : stationname
# data/time : receiver snrs
# and then build a set like this:
inputdata = {
    "John": {"Groningen": 50, "Leeuwarden": 43, "Smilde": 29, "Goes": 12, "Ignored": 123},
    "Peter": {"Groningen": 5, "Leeuwarden": 90, "Smilde": 1, "Goes": 3, "Ignored": 123}
}

stations = list(map(stationToTensor, inputdata.keys()))
signals = list(map(signalToTensor, inputdata.values()))

dataset = tf.data.Dataset.from_tensor_slices((signals, stations))
print(dataset)
