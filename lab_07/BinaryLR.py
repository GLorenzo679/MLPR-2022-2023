import numpy as np
import scipy
import sklearn.datasets


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


class logRegClass:
    def __init__(self, DTR, LTR, l):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l

    def logreg_obj(self, v):
        w, b = v[0:-1], v[-1]

        return (0.5 * self.l * np.linalg.norm(np.dot(w, w))) + (
            self.LTR * np.logaddexp(0, (np.dot(w.T, self.DTR) + b))
            + (1 - self.LTR) * np.logaddexp(0, (-np.dot(w.T, self.DTR) - b))
        ).sum() / self.DTR.shape[1]


def score(v, DTE):
    w_star, b_star = v[0:-1], v[-1]
    score = []

    for i in range(DTE.shape[1]):
        score.append(np.dot(w_star.T, DTE[:, i]) + b_star)

    return np.array(score)


def evaluate(score_array, LTE):
    LP = np.array([1 if x > 0 else 0 for x in score_array])

    matched = np.array([True if x1 == x2 else False for x1, x2 in zip(LP, LTE)])
    error_rate = matched.mean()

    return 100 * (error_rate)


def main():
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    lambda_array = [0.000001, 0.001, 0.1, 1]

    for l in lambda_array:
        # initialize starting point (dim = dim_w + dim_b)
        x0 = np.zeros(DTR.shape[0] + 1)
        # initialize logRegClass object
        log_reg_obj = logRegClass(DTR, LTR, l)

        # minimize objective function
        v, J, info = scipy.optimize.fmin_l_bfgs_b(log_reg_obj.logreg_obj, x0, approx_grad=True)

        # compute score and error rate
        score_array = score(v, DTE)
        error_rate = evaluate(score_array, LTE)

        print(f"Error rate for lambda={l}:\t{error_rate:.1f}%")


if __name__ == "__main__":
    main()
