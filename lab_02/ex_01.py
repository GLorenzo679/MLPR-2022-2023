import csv
import os

import matplotlib.pyplot as plt
import numpy as np

PATH = os.path.abspath(os.path.dirname(__file__))


def load(filepath):
    data_matrix = []
    class_array = []

    with open(filepath) as csv_file:
        reader = csv.reader(csv_file, delimiter=",")

        for row in reader:
            data_matrix.append(np.array(row[0:4], dtype=np.float32))

            if row[4] == "Iris-setosa":
                class_array.append(0)
            elif row[4] == "Iris-versicolor":
                class_array.append(1)
            else:
                class_array.append(2)

    return np.vstack(data_matrix).T, np.array(class_array, dtype=np.int32)


def plot_hist(matrix, label):
    M0 = matrix[:, label == 0]
    M1 = matrix[:, label == 1]
    M2 = matrix[:, label == 2]

    x_labels = {0: "Sepal length", 1: "Sepal width", 2: "Petal length", 3: "Petal width"}

    for i in range(4):
        plt.figure()

        plt.hist(M0[i, :], bins=10, density=True, alpha=0.4, label="Iris-Setosa")
        plt.hist(M1[i, :], bins=10, density=True, alpha=0.4, label="Iris-Versicolor")
        plt.hist(M2[i, :], bins=10, density=True, alpha=0.4, label="Iris-Virginica")

        plt.xlabel(x_labels[i])
        plt.legend(loc="upper right")
    plt.show()


def plot_scatter(matrix, label):
    M0 = matrix[:, label == 0]
    M1 = matrix[:, label == 1]
    M2 = matrix[:, label == 2]

    x_labels = {0: "Sepal length", 1: "Sepal width", 2: "Petal length", 3: "Petal width"}

    for i in range(4):
        for j in range(4):
            if i != j:
                plt.figure()

                plt.scatter(M0[i, :], M0[j, :], label="Iris-Setosa")
                plt.scatter(M1[i, :], M1[j, :], label="Iris-Versicolor")
                plt.scatter(M2[i, :], M2[j, :], label="Iris-Virginica")

                plt.xlabel(x_labels[i])
                plt.ylabel(x_labels[j])
                plt.legend()
    plt.show()


def mean(matrix):
    mu = 0

    for i in range(matrix.shape[1]):
        mu += matrix[:, i : i + 1]

    mu /= float(matrix.shape[1])


def main():
    matrix, class_array = load(PATH + "/data/iris.csv")

    plot_hist(matrix, class_array)
    plot_scatter(matrix, class_array)

    # mean(matrix)  --> slower, use numpy function instead

    # remove the mean from all points
    matrix_centered = matrix - matrix.mean(1).reshape(matrix.shape[0], 1)

    plot_scatter(matrix_centered, class_array)


if __name__ == "__main__":
    main()
