import os
from operator import itemgetter


def main():
    filepath = os.getcwd()
    athlete_list = []
    country_list = {}

    with open(filepath + "\ex_01\score.txt", "r") as f:
        lines = f.readlines()

        for line in lines:
            sum = 0

            # parse line and do operations
            fields = line.rstrip().split()
            athlete = {
                "name": fields[0] + " " + fields[1],
                "country": fields[2],
                "score": fields[3:],
            }
            athlete["score"].remove(min(athlete["score"]))
            athlete["score"].remove(max(athlete["score"]))

            for e in athlete["score"]:
                sum += float(e)

            athlete["score"] = sum

            athlete_list.append(athlete)

            # calculate country total score
            if fields[2] in country_list.keys():
                country_list[fields[2]] += sum
            else:
                country_list[fields[2]] = sum

    # select top 3 athletes based on score
    top_3 = sorted(athlete_list, key=itemgetter("score"), reverse=True)[:3]

    for i in range(len(top_3)):
        name = top_3[i]["name"]
        score = top_3[i]["score"]
        print(f"{i + 1}: {name} -- Score: {score}")

    # select best country based on score
    best_country = sorted(country_list.items(), key=lambda x: x[1], reverse=True).pop(0)
    country = best_country[0]
    score = best_country[1]
    print(f"\n{country} Total score: {score}")


if __name__ == "__main__":
    main()
