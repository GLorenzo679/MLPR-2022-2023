import os
import calendar

def main():
    filepath = os.getcwd()
    city_count = {}
    month_count = {}

    with open(filepath + "/ex_03/people.txt", "r") as f:
        lines = f.readlines()

        for line in lines:
            fields = line.rstrip().split()
            month = calendar.month_name[int(fields[3].split("/")[1])]
            
            if fields[2] in city_count.keys():
                city_count[fields[2]] += 1
            else:
                city_count[fields[2]] = 1

            if month in month_count.keys():
                month_count[month] += 1
            else:
                month_count[month] = 1
                
    print("Birth per city:")

    for k, v in city_count.items():
        print(f"{k}: {v}")

    print("\nBirths per month:")

    for k, v in month_count.items():
        print(f"{k}: {v}")

    average = sum(city_count.values())/len(city_count)

    print(f"\nAverage number of births: {average}")

if __name__ == "__main__":
    main() 