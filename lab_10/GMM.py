import os

import numpy as np
from GMM_load import load_gmm
from scipy.special import logsumexp

PATH = os.path.abspath(os.path.dirname(__file__))


def vcol(v):
    return v.reshape(v.shape[0], 1)


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


def logpdf_GMM(X, gmm):
    S = np.zeros((len(gmm), X.shape[1]))

    # compute joint log-density for each Gaussian component of the GMM
    for g in range(len(gmm)):
        w = gmm[g][0]
        mu = vcol(gmm[g][1])
        C = gmm[g][2]

        # compute overall ll, by doing a weighted sum over every gaussian density
        # ll = logpdf_GAU + log(w)
        S[g] = logpdf_GAU_ND_fast(X, mu, C)
        S[g, :] += np.log(w)

    log_density = logsumexp(S, axis=0)

    return log_density


def main():
    dataset_4D = np.load(PATH + "/data/GMM_data_4D.npy")
    reference_GMM = load_gmm(PATH + "/data/GMM_4D_3G_init.json")

    log_density = logpdf_GMM(dataset_4D, reference_GMM)

    log_density_sol = np.load(PATH + "/data/GMM_4D_3G_init_ll.npy")
    print(f"Log joint density error (sol - mine): {np.abs(log_density_sol - log_density).max()}")


if __name__ == "__main__":
    main()
