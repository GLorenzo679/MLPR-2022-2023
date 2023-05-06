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


def plot_scatter(matrix, label):
    M0 = matrix[:, label == 0]
    M1 = matrix[:, label == 1]
    M2 = matrix[:, label == 2]

    x_labels = {0: "Sepal length", 1: "Sepal width", 2: "Petal length", 3: "Petal width"}

    i = 0
    j = 1

    plt.scatter(M0[i, :], M0[j, :], label="Iris-Setosa")
    plt.scatter(M1[i, :], M1[j, :], label="Iris-Versicolor")
    plt.scatter(M2[i, :], M2[j, :], label="Iris-Virginica")

    plt.xlabel(x_labels[i])
    plt.ylabel(x_labels[j])
    plt.legend()
    plt.show()


def PCA(D, m):
    # remove the mean from all points
    Dc = D - D.mean(1).reshape(D.shape[0], 1)

    # calculate covariance matrix
    C = np.dot(Dc, Dc.T) / (D.shape[0])

    # compute eigenvalues and eigenvectors sorted
    s, U = np.linalg.eigh(C)

    # alternative method to calculate eigenvalues and eigenvectors
    # only possible because covariance matrix is semi-definite positive
    # U, s, Vh = np.linalg.svd(C)

    P = U[:, ::-1][:, 0:m]

    # project the data onto the principal components
    DP = np.dot(P.T, D)

    return DP


def main():
    D, class_array = load(PATH + "/data/iris.csv")
    DP = PCA(D, 2)
    plot_scatter(DP, class_array)


if __name__ == "__main__":
    main()
