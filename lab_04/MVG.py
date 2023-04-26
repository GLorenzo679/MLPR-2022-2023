import os
import numpy as np
import matplotlib.pyplot as plt


def vrow(v):
    return v.reshape(1, v.shape[0])


def logpdf_GAU_ND_1Sample(x, mu, C):
    # x data centered
    xc = x - mu
    # size of x
    M = x.shape[0]
    const = -0.5 * M * np.log(2 * np.pi)
    # compute log determinant of covariance matrix -> log|C|
    logdet_C = np.linalg.slogdet(C)[1]
    # inverse of covariance matrix -> precision matrix
    L = np.linalg.inv(C)
    # calculate x_T*(1/C*xc)
    # (1, 1000)*[(1000,1000)(1000,1)]
    v = np.dot(xc.T, np.dot(L, xc)).ravel()

    return const -0.5 * logdet_C -0.5 * v 


def logpdf_GAU_ND(X, mu, C):
    Y = []

    for i in range(X.shape[1]):
        Y.append(logpdf_GAU_ND_1Sample(X[:, i:i+1], mu, C))

    return np.array(Y).ravel()


def logpdf_GAU_ND_fast(X, mu, C):
    XC = X - mu
    M = X.shape[0]
    const = -0.5 * M * np.log(2 * np.pi)
    logdet = np.linalg.slogdet(C)[1]
    L = np.linalg.inv(C)
    v = (XC * np.dot(L, XC)).sum(0)

    return const -0.5 * logdet -0.5 * v 

def main():
    plt.figure()
    
    # data
    XPlot = np.linspace(-8, 12, 1000)
    # mean (np 1x1 array)
    mu = np.ones((1,1)) * 1.0
    # covariance matrix (np 1x1 array)
    C = np.ones((1,1)) * 2.0

    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND_fast(vrow(XPlot), mu, C)))
    plt.show()

    pdfSol = np.load(os.getcwd() + "/lab_04/data/llGAU.npy")
    pdfGau = logpdf_GAU_ND(vrow(XPlot), mu, C)

    print(np.abs(pdfSol - pdfGau).max())


if __name__ == "__main__":
    main()