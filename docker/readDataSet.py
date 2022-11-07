#!python
#
# Library to read a CSV file and cough up an x/y trainingsset for Tensorflow
#
import csv


def normalizeWeekday(date):
    # Take a day of the week and normalize it to 0.0 to 1.0
    return 0.3

def normalizeTimeOfDay(date):
    # Take a time of day from the date and normalize it to 0.0 to 1.0
    return 0.5

def normalizeRxValue(value):
    return 0.6

# Builds transforms a list of station names and the active station
# into a probability array:
# builsStationArray(["A", "B", "C"], "B") -> [0.0, 1.0, 0.0] 
def buildStationArray(stations, name):
    return list(map(lambda x: 1.0 if x == name else 0.0, stations))

# Flattens an array of arrays into a 1 dimensional array
def flatten(array2d):
    return [value for ar in array2d for value in ar]


# Returns an x/y trainingsset where:
# x is an array containing arrays of input values
# y is an array containing arrays of output probabilities (classification array)
# stations is an array containing the names belonging to the output probabilities
def readTrainingSet(filename):
    header = ""
    stations=[]
    x=[]
    y=[]

    # Find all stations to detect
    with open(filename) as f:
        csv_reader = csv.reader(f, delimiter=';')
        header = next(csv_reader)
        stationColumn = header.index("station")
        for line in csv_reader:
            if not line[stationColumn] in stations:
                stations.append(line[stationColumn])

    # Read all SNR values and map them to station probabilities
    with open(filename) as f:
        csv_reader = csv.reader(f, delimiter=';')
        header = next(csv_reader)

        for line in csv_reader:
            weekday = normalizeWeekday(line[0])
            hour = normalizeTimeOfDay(line[0])
            rx_hvs = normalizeRxValue(line[header.index("Rx_Hvs")])
            rx_bm = normalizeRxValue(line[header.index("Rx_Bm")])

            x.append([weekday, hour, rx_hvs, rx_bm])
            y.append(buildStationArray(stations, line[header.index("station")]))

    return {
        "x" : x,
        "y" : y,
        "stations" : stations
    }

# Run if this is not included as library
if __name__ == '__main__':
    data = readTrainingSet("data.csv")
    print(data["x"], flatten(data["x"]))
    print(data["y"], flatten(data["y"]))
    print(data["stations"])
