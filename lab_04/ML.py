import os

import matplotlib.pyplot as plt
import numpy as np


def vrow(v):
    return v.reshape(1, v.shape[0])


def logpdf_GAU_ND_fast(X, mu, C):
    XC = X - mu
    M = X.shape[0]
    const = -0.5 * M * np.log(2 * np.pi)
    logdet = np.linalg.slogdet(C)[1]
    L = np.linalg.inv(C)
    v = (XC * np.dot(L, XC)).sum(0)

    return const - 0.5 * logdet - 0.5 * v


def loglikelihood(XND, mu_ML, C_ML):
    pdfGAU = logpdf_GAU_ND_fast(XND, mu_ML, C_ML)
    # sum of the log-density computed over all the samples
    return pdfGAU.sum()


def main():
    XPlot = np.linspace(-8, 12, 1000)
    X1D = np.load(os.getcwd() + "/lab_04/data/X1D.npy")
    XND = np.load(os.getcwd() + "/lab_04/data/XND.npy")

    # N dimensional case
    XNDc = XND - XND.mean(1).reshape(XND.shape[0], 1)
    # compute mean
    muN_ML = XND.mean(1).reshape(XND.shape[0], 1)
    # compute covariance matrix
    CN_ML = np.dot(XNDc, XNDc.T) / XNDc.shape[1]

    print(f"ND mean:\n{muN_ML}\n")
    print(f"ND covariance matrix:\n{CN_ML}\n")

    ll = loglikelihood(XND, muN_ML, CN_ML)

    print(f"ND log likelihood: {ll}\n")

    # 1 dimensional case
    X1Dc = X1D - X1D.mean(1).reshape(X1D.shape[0], 1)
    # compute mean
    mu1_ML = X1D.mean(1).reshape(X1D.shape[0], 1)
    # compute covariance matrix
    C1_ML = np.dot(X1Dc, X1Dc.T) / X1Dc.shape[1]

    print(f"ND mean:\n{mu1_ML}\n")
    print(f"ND covariance matrix:\n{C1_ML}\n")

    ll = loglikelihood(X1D, mu1_ML, C1_ML)

    print(f"1D log likelihood: {ll}\n")

    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND_fast(vrow(XPlot), mu1_ML, C1_ML)))
    plt.show()


if __name__ == "__main__":
    main()
