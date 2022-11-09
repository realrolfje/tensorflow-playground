#!python
#
# Library to read a CSV file and cough up an x/y trainingsset for Tensorflow
#
import csv
from datetime import datetime, timedelta

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def toDate(dateString):
    cleaned = (dateString + "000")[0:dateString.find(".")+4]
    return datetime.fromisoformat(cleaned)


def normalizeWeekday(date):
    return date.weekday()/6.0

def normalizeTimeOfDay(date):
    t = date.time()
    seconds = timedelta(hours=t.hour, minutes=t.minute,seconds=t.second).total_seconds()
    return seconds / 86399.0

def normalizeRxValue(value):
    f = None
    if isinstance(value, int):
        f = float(value)
    if isinstance(value, str) and value != "":
        f = float(value)
    if f is None:
        return 0.0

    try:
        return clamp(f, 0.0, 100.0)/100.0
    except:
        raise Exception("Can not normalize rx value '%s'" % value)

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

    receivers = ["Rx_Alm", "Rx_Bd", "Rx_Bo", "Rx_Ehv", "Rx_Ems", "Rx_Hvs", "Rx_Kp",
        "Rx_Lls", "Rx_Lpk", "Rx_Lw", "Rx_Nm", "Rx_Nsw", "Rx_PJ2", "Rx_Rie", "Rx_Rt",
        "Rx_Sch", "Rx_Std", "Rx_Vli", "Rx_Vli_UK", "Rx_Zze"]

    # Find all stations to detect
    with open(filename) as f:
        csv_reader = csv.reader(f, delimiter=';')
        header = next(csv_reader)
        # stationColumn = header.index("station")
        # for line in csv_reader:
        #     if not line[stationColumn] in stations:
        #         stations.append(line[stationColumn])

    # Read all SNR values and map them to station probabilities
    with open(filename) as f:
        csv_reader = csv.reader(f, delimiter=';')
        header = next(csv_reader)

        for line in csv_reader:
            qsodatetime = toDate(line[0])
            weekday = normalizeWeekday(qsodatetime)
            hour = normalizeTimeOfDay(qsodatetime)
            signals = [normalizeRxValue(line[header.index(s)]) for s in receivers]
            x.append([weekday, hour] + signals)
            # y.append(buildStationArray(stations, line[header.index("station")]))

    return {
        "x" : x,
        "y" : y,
        "stations" : stations
    }

# Run if this is not included as library
if __name__ == '__main__':
    data = readTrainingSet("20221107-export/receiver-signals.csv")
    for i in range(30):
        print(data["x"][i])

    # print(data["x"], flatten(data["x"]))
    # print(data["y"], flatten(data["y"]))
    # print(data["stations"])
