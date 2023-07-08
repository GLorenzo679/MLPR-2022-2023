import os

import numpy as np
import sklearn.datasets
from scipy.optimize import fmin_l_bfgs_b

PATH = os.path.abspath(os.path.dirname(__file__))


def vcol(v):
    return v.reshape(v.shape[0], 1)


def vrow(v):
    return v.reshape(1, v.shape[0])


def load_iris_binary():
    D, L = (sklearn.datasets.load_iris()["data"].T, sklearn.datasets.load_iris()["target"])

    D = D[:, L != 0]  # We remove setosa from D
    L = L[L != 0]  # We remove setosa from L
    L[L == 2] = 0  # We assign label 0 to virginica (was label 2)

    return D, L


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)

    np.random.seed(seed)

    idx = np.random.permutation(D.shape[1])

    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)


def obj_fun_gradient(alpha, H, D):
    grad_L = np.dot(H, alpha) - np.ones(D.shape[1])

    return np.reshape(grad_L, (D.shape[1],))


def obj_fun(alpha, H, D):
    L = 0.5 * np.dot(alpha.T, np.dot(H, alpha)) - np.dot(alpha, np.ones(np.shape(alpha)[0]))

    return L, obj_fun_gradient(alpha, H, D)


def compute_duality_gap(w_star, z, D, C, L_dual):
    J_primal = 0.5 * np.linalg.norm(w_star) ** 2 + C * np.sum(np.maximum(0, 1 - z * np.dot(w_star.T, D)))

    print(f"Primal loss: {J_primal}")

    return J_primal + L_dual


def evaluate(score_array, LTE):
    z = np.where(LTE == 0, 1, -1)

    matched = (np.where(score_array > 0, 1, -1)) == z
    error_rate = matched.mean()

    return 100 * (error_rate)


def SVM_kernel(DTR, H, C, z):
    xopt, fopt, info = fmin_l_bfgs_b(
        obj_fun,
        np.zeros(DTR.shape[1]),
        approx_grad=False,
        factr=1.0,
        bounds=[(0, C) for _ in range(DTR.shape[1])],
        args=(H, DTR),
    )

    print(f"Dual loss: {-fopt}")

    # return alpha star
    return xopt


def SVM_kernel_score(DTR, DTE, alpha_star, z, kernel_func, d, c, gamma, K):
    scores = np.zeros(DTE.shape[1])

    for i in range(DTE.shape[1]):
        for j in range(DTR.shape[1]):
            if alpha_star[j] <= 0:
                continue

            if kernel_func == polynomial_kernel:
                scores[i] += alpha_star[j] * z[j] * (kernel_func(DTR[:, j], DTE[:, i], c, d, K))
            elif kernel_func == RBF_kernel:
                scores[i] += alpha_star[j] * z[j] * (kernel_func(DTR[:, j], DTE[:, i], gamma, K))

    return scores


def polynomial_kernel(X1, X2, c, degree, K):
    return (np.dot(X1.T, X2) + c) ** degree + K**2


def RBF_kernel(X1, X2, gamma, K):
    if np.array_equal(X1, X2):
        D = X1
        n = D.shape[1]
        diff_mat = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                diff_mat[i, j] = (np.linalg.norm(D[:, i] - D[:, j])) ** 2
        return np.exp(-gamma * diff_mat) + K**2
    else:
        return np.exp(-gamma * np.linalg.norm(X1 - X2) ** 2) + K**2


def main():
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    K_array = [0, 1]

    for _ in range(2):
        # Polynomial kernel
        for c in [0, 1]:
            print("Poly:")
            print(f"c = {c}:\n")
            d = 2

            for K in K_array:
                print(f"K = {K}: ")

                z = 2 * LTR - 1

                H = polynomial_kernel(DTR, DTR, c, d, K) * vrow(z) * vcol(z)
                C = 1

                alpha_star = SVM_kernel(DTR, H, C, z)
                score_array = SVM_kernel_score(DTR, DTE, alpha_star, z, polynomial_kernel, d, c, None, K)
                error_rate = evaluate(score_array, LTE)
                print(f"Error rate: {error_rate:.1f}%\n")
                print("-" * 50 + "\n")

        for K in K_array:
            print("RBF:")
            print(f"K = {K}: ")

            for gamma in [1, 10]:
                print(f"gamma = {gamma}:\n")

                z = 2 * LTR - 1

                H = RBF_kernel(DTR, DTR, gamma, K) * vrow(z) * vcol(z)
                C = 1

                alpha_star = SVM_kernel(DTR, H, C, z)
                score_array = SVM_kernel_score(DTR, DTE, alpha_star, z, RBF_kernel, None, None, gamma, K)
                error_rate = evaluate(score_array, LTE)
                print(f"Error rate: {error_rate:.1f}%\n")
                print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
