import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


def vcol(v):
    return v.reshape(v.shape[0], 1)


def vrow(v):
    return v.reshape(1, v.shape[0])


def load(filepath):
    data_matrix = []
    class_array = []

    with open(filepath) as csv_file:
        reader = csv.reader(csv_file, delimiter = ',')
        
        for row in reader:
            data_matrix.append(np.array(row[0:4], dtype = np.float32))

            if(row[4] == "Iris-setosa"):
                class_array.append(0)
            elif(row[4] == "Iris-versicolor"):
                class_array.append(1)
            else:
                class_array.append(2)

    return np.vstack(data_matrix).T, np.array(class_array, dtype = np.int32)


def plot_scatter(matrix, label):
    M0 = matrix[:, label == 0]
    M1 = matrix[:, label == 1]
    M2 = matrix[:, label == 2]

    x_labels = {
        0 : "Sepal length",
        1 : "Sepal width",
        2 : "Petal length",
        3 : "Petal width"
    }

    i = 0
    j = 1

    plt.scatter(M0[i, :], M0[j, :], label = "Iris-Setosa")
    plt.scatter(M1[i, :], M1[j, :], label = "Iris-Versicolor")
    plt.scatter(M2[i, :], M2[j, :], label = "Iris-Virginica")

    plt.xlabel(x_labels[i])
    plt.ylabel(x_labels[j])
    plt.legend()
    plt.show()


def SbSw(D, L):
    SW = 0
    SB = 0

    # calculate mean of all dataset
    mean = vcol(D.mean(1))

    for i in range(3):
        # select data of each class
        D_class = D[:, L == i]
        # calculate mean of each class
        mean_class = D_class.mean(1).reshape(D_class.shape[0], 1)
        # calculate between class covariance matrix
        SB += D_class.shape[1] * np.dot((mean_class-mean), (mean_class-mean).T)
        # calculate the within class covariance matrix
        SW += np.dot((D_class - mean_class), (D_class - mean_class).T)
    
    SB /= D.shape[1]
    SW /= D.shape[1]

    return SB, SW


def LDA1(D, L, m):
    # compute convolution matrices
    SB, SW =  SbSw(D, L)

    # solve the generalized eigenvelue problem
    # only possible beacuse Sw is positive defined
    s, U = scipy.linalg.eigh(SB, SW)
    return U[:, ::-1][:, 0:m]


def LDA2(D, L, m):
    # compute convolution matrices
    SB, SW =  SbSw(D, L)

    # Solving the eigenvalue problem by joint diagonalization
    U, s, _ = np.linalg.svd(SW)

    P1 = np.dot(U * vrow(1.0/(s**0.5)), U.T)
    SBTilde = np.dot(P1, np.dot(SB, P1.T))
    U, _, _ = np.linalg.svd(SBTilde)

    P2 = U[:, 0:m]
    return np.dot(P1.T, P2)


def main():
    filepath = os.getcwd() + "/data/iris.csv"
    test_out_data = np.load(os.getcwd() + "/data/IRIS_LDA_matrix_m2.npy")

    D, class_array = load(filepath)

    m = 2

    W1 = LDA1(D, class_array, m)
    W2 = LDA2(D, class_array, m)*-1


if __name__ == "__main__":
    main()