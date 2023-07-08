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


def SVM_linear(DTR, H, C, z):
    xopt, fopt, info = fmin_l_bfgs_b(
        obj_fun,
        np.zeros(DTR.shape[1]),
        approx_grad=False,
        factr=1.0,
        bounds=[(0, C) for _ in range(DTR.shape[1])],
        args=(H, DTR),
    )

    z = np.reshape(z, z.shape[0])
    w_star = np.dot(DTR, z * xopt)

    # w_star = np.sum(np.dot(DTR, xopt * z), axis=1)
    duality_gap = compute_duality_gap(w_star, z, DTR, C, fopt)
    print(f"Dual loss: {fopt}")
    print(f"Duality gap: {duality_gap}")

    return w_star


def main():
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    K_array = [1, 10]

    for K in K_array:
        print(f"K = {K}: ")
        DTR_extended = np.vstack((DTR, K * np.ones(DTR.shape[1])))
        DTE_extended = np.vstack((DTE, K * np.ones(DTE.shape[1])))

        G = np.dot(DTR_extended.T, DTR_extended)
        z = 2 * LTR - 1
        # exploit broadcasting to compute H
        H = G * vrow(z) * vcol(z)

        C_array = [0.1, 1, 10]

        for C in C_array:
            print(f"C = {C}:")
            w_star = SVM_linear(DTR_extended, H, C, z)
            score_array = np.dot(w_star.T, DTE_extended)
            error_rate = evaluate(score_array, LTE)
            print(f"Error rate: {error_rate:.1f}%\n")
            print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
