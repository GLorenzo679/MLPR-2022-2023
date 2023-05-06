import math
import os
import sys

PATH = os.path.abspath(os.path.dirname(__file__))


def main(argv):
    filepath = os.getcwd()
    bus_list = []

    with open(PATH + "/public_transportation.txt", "r") as f:
        lines = f.readlines()

        for line in lines:
            fields = line.rstrip().split()
            bus_list.append(
                {
                    "busId": fields[0],
                    "lineId": fields[1],
                    "x": fields[2],
                    "y": fields[3],
                    "time": fields[4],
                }
            )

    if argv[1] == "-b":
        distance = 0
        x = 0
        y = 0
        busId = argv[2]

        for bus in bus_list:
            if bus["busId"] == busId:
                if x == 0 and y == 0:
                    x = int(bus["x"])
                    y = int(bus["y"])
                else:
                    distance += math.sqrt((int(bus["x"]) - x) ** 2 + (int(bus["y"]) - y) ** 2)
                    x = int(bus["x"])
                    y = int(bus["y"])

        print(f"{busId} - Total Distance: {distance}")
    elif argv[1] == "-l":
        distance = 0
        tot_time = 0
        x = 0
        y = 0
        time = 0
        lineId = argv[2]

        for bus in bus_list:
            if bus["lineId"] == lineId:
                if (x == 0 and y == 0 and time == 0) or int(bus["time"]) < time:
                    x = int(bus["x"])
                    y = int(bus["y"])
                    time = int(bus["time"])
                    busId = bus["busId"]
                else:
                    distance += math.sqrt((int(bus["x"]) - x) ** 2 + (int(bus["y"]) - y) ** 2)
                    tot_time += int(bus["time"]) - time
                    x = int(bus["x"])
                    y = int(bus["y"])
                    time = int(bus["time"])

        print(f"{lineId} - Avg Speed: {distance/tot_time}")


if __name__ == "__main__":
    main(sys.argv[:])
