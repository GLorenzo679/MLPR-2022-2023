import numpy as np
import sklearn.datasets


def vcol(v):
    return v.reshape(v.shape[0], 1)


def vrow(v):
    return v.reshape(1, v.shape[0])


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


def evaluate_classifier(predictions, labels):
    # compute boolean array, true if prediction == eval label else false
    matched = np.array([True if x1 == x2 else False for x1, x2 in zip(predictions, labels)])

    # sum totale number of True (correct predictions) and divide by number of samples
    accuracy = matched.sum() / predictions.size
    error_rate = 1.0 - accuracy

    return 100 * accuracy, 100 * error_rate


def logpdf_GAU_ND_fast(X, mu, C):
    XC = X - mu
    M = X.shape[0]
    const = -0.5 * M * np.log(2 * np.pi)
    logdet = np.linalg.slogdet(C)[1]
    L = np.linalg.inv(C)
    v = (XC * np.dot(L, XC)).sum(0)

    return const - 0.5 * logdet - 0.5 * v


def score_matrix(DTV, mean_array, cov_array, n_label):
    S = []

    for i in range(n_label):
        if cov_array.ndim > 2:
            fcond = np.exp(logpdf_GAU_ND_fast(DTV, mean_array[i], cov_array[i]))
        else:
            fcond = np.exp(logpdf_GAU_ND_fast(DTV, mean_array[i], cov_array))

        S.append(vrow(fcond))

    return np.vstack(S)
