import os

import numpy as np
from scipy.special import logsumexp

PATH = os.path.abspath(os.path.dirname(__file__))


def vcol(v):
    return v.reshape(v.size, 1)


def vrow(v):
    return v.reshape(1, v.size)


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

    for g in range(len(gmm)):
        w = gmm[g][0]
        mu = vcol(gmm[g][1])
        C = gmm[g][2]

        S[g] = logpdf_GAU_ND_fast(X, mu, C)
        S[g, :] += np.log(w)

    log_density = logsumexp(S, axis=0)

    return S, log_density


def MVG_log_classifier(S, log_density):
    log_S_marginal = vrow(log_density)
    log_S_post = S - log_S_marginal

    return np.exp(log_S_post)


def E_step(X, gmm):
    S, log_density = logpdf_GMM(X, gmm)
    # calculate posterior = responsibility
    S_post = MVG_log_classifier(S, log_density)

    return S_post, log_density


def M_step(gamma, X, psi, diagCov, tiedCov):
    Z_list = []

    mu_list = []
    cov_list = []
    w_list = []

    tied_cov_matrix = np.zeros((X.shape[0], X.shape[0]))

    for gamma_g in gamma:
        Z_g = np.sum(gamma_g)
        Z_list.append(Z_g)
        F_g = vcol(np.dot(vrow(gamma_g), X.T))
        S_g = np.dot(vrow(gamma_g) * X, X.T)

        mu_g = F_g / Z_g
        mu_list.append(mu_g)

        C_g = S_g / Z_g - np.dot(mu_g, mu_g.T)

        if diagCov:
            # only keep diagonal elements of C_g
            C_g *= np.eye(C_g.shape[0])
        if tiedCov:
            tied_cov_matrix += Z_g * C_g

        cov_list.append(C_g)

    if tiedCov == True:
        tied_cov_matrix /= X.shape[1]

    for i in range(len(cov_list)):
        U, s, _ = np.linalg.svd(cov_list[i])
        s[s < psi] = psi
        cov_list[i] = np.dot(U, vcol(s) * U.T)

    w_list = [Z_g / np.sum(Z_list) for Z_g in Z_list]

    new_gmm = [(w_list[g], mu_list[g], cov_list[g]) for g in range(len(gamma))]

    return new_gmm


def EM_GMM(
    X,
    gmm,
    psi,
    diagCov,
    tiedCov,
    threshold=1e-6,
):
    new_gmm = gmm
    avg_log_ll = 0

    while True:
        old_avg_log_ll = avg_log_ll

        # E-step
        gamma, log_ll = E_step(X, new_gmm)
        avg_log_ll = np.mean(log_ll)

        if np.abs(avg_log_ll - old_avg_log_ll) < threshold:
            print(f"Converged at avg_log_ll: {avg_log_ll}")
            return new_gmm

        # M-step
        new_gmm = M_step(gamma, X, psi, diagCov, tiedCov)


def mean_cov_estimate(D):
    mu = D.mean(1).reshape(D.shape[0], 1)

    Dc = D - D.mean(1).reshape(D.shape[0], 1)
    C = np.dot((D - Dc), (D - Dc).T) / (D.shape[1] - 1)

    return mu, C


def GMM_split(gmm, alpha=0.1):
    new_gmm = []

    for g in range(len(gmm)):
        U, s, Vh = np.linalg.svd(gmm[g][2])
        d = U[:, 0] * s[0] ** 0.5 * alpha

        mu = gmm[g][1]
        C = gmm[g][2]

        mu1 = mu + vcol(d)
        mu2 = mu - vcol(d)
        C1 = C
        C2 = C

        new_gmm.append((gmm[g][0] / 2, mu1, C1))
        new_gmm.append((gmm[g][0] / 2, mu2, C2))

    return new_gmm


def LBG(X, gmm, psi, diagCov, tiedCov):
    opt_gmm = gmm

    for _ in range(2):
        opt_gmm = EM_GMM(X, GMM_split(opt_gmm), psi, diagCov, tiedCov)

    return opt_gmm


def main():
    dataset_4D = np.load(PATH + "/data/GMM_data_4D.npy")

    psi = 0.01
    mu, C = mean_cov_estimate(dataset_4D)

    U, s, _ = np.linalg.svd(C)
    s[s < psi] = psi
    C_adjusted = np.dot(U, vcol(s) * U.T)

    GMM_1 = [(1.0, mu, C_adjusted)]

    opt_gmm = LBG(dataset_4D, GMM_1, psi, diagCov=False, tiedCov=True)


if __name__ == "__main__":
    main()
