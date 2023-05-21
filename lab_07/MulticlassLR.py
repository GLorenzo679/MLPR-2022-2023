import sys

import numpy as np
import scipy.optimize
import scipy.special
import sklearn.datasets


def vcol(v):
    return v.reshape(v.shape[0], 1)


def load_iris():
    D, L = (
        sklearn.datasets.load_iris()["data"].T,
        sklearn.datasets.load_iris()["target"],
    )

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


class logRegModel:
    def __init__(self, DTR, LTR, K, l):
        self.DTR = DTR
        self.LTR = LTR
        self.K = K
        self.l = l

    def logreg_obj(self, v):
        W, b = v[0 : -self.K], vcol(v[-self.K :])
        W = np.reshape(W, (self.DTR.shape[0], self.K))

        S = np.dot(W.T, self.DTR) + b
        Y_log = S - scipy.special.logsumexp(S, axis=0)

        # T should be 1 of k encoding of labels -> one hot encoding
        T = np.array([self.LTR == i for i in range(self.K)]).astype(int)

        return (0.5 * self.l * (W * W).sum()) - (T * Y_log).sum() / self.DTR.shape[1]


def score(v, DTE, K):
    W_star, b_star = v[0:-K], v[-K:]
    W_star = np.reshape(W_star, (DTE.shape[0], K))
    score = []

    for i in range(DTE.shape[1]):
        score.append(np.dot(W_star.T, DTE[:, i]) + b_star)

    return np.array(score).T


def evaluate(score_array, LTE):
    LP = np.argmax(score_array, axis=0)

    matched = np.array([True if x1 == x2 else False for x1, x2 in zip(LP, LTE)])
    accuracy = matched.mean()

    return 100 * (1 - accuracy)


def main():
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    # count number of classes
    K = len(np.unique(LTE))

    lambda_array = [0.000001, 0.001, 0.1, 1]

    for l in lambda_array:
        # initialize starting point (dim = dim_W + dim_b)
        x0 = np.zeros(DTR.shape[0] * K + K)
        # initialize logRegClass object
        log_reg_obj = logRegModel(DTR, LTR, K, l)

        # minimize objective function
        v, J, info = scipy.optimize.fmin_l_bfgs_b(log_reg_obj.logreg_obj, x0, approx_grad=True)

        # compute score and error rate
        score_array = score(v, DTE, K)
        error_rate = evaluate(score_array, LTE)

        print(f"Error rate for lambda={l}:\t{error_rate:.1f}%")


if __name__ == "__main__":
    main()
