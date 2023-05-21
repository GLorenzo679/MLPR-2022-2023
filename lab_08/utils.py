import numpy as np
import scipy
import sklearn.datasets


def vcol(v):
    return v.reshape(v.shape[0], 1)


def vrow(v):
    return v.reshape(1, v.shape[0])


def mcol(m):
    return m.reshape((m.size, 1))


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
    matched = np.array([True if x1 == x2 else False for x1, x2 in zip(predictions, labels)])
    accuracy = matched.mean()
    error_rate = 1.0 - accuracy

    return 100 * accuracy, 100 * error_rate


class MVGLogClassifier:
    def __init__(self, prior):
        self.mean_array = None
        self.cov_array = None
        self.prior = prior

    def __logpdf_GAU_ND_fast__(self, X, mu, C):
        XC = X - mu
        M = X.shape[0]
        const = -0.5 * M * np.log(2 * np.pi)
        logdet = np.linalg.slogdet(C)[1]
        L = np.linalg.inv(C)
        v = (XC * np.dot(L, XC)).sum(0)

        return const - 0.5 * logdet - 0.5 * v

    def __score_matrix__(self, DTV, n_label):
        S = []

        for i in range(n_label):
            if self.cov_array.ndim > 2:
                fcond = np.exp(self.__logpdf_GAU_ND_fast__(DTV, self.mean_array[i], self.cov_array[i]))
            else:
                fcond = np.exp(self.__logpdf_GAU_ND_fast__(DTV, self.mean_array[i], self.cov_array))

            S.append(vrow(fcond))

        return np.vstack(S)

    def train(self, D, L):
        mean_array = []
        cov_array = []

        for i in range(3):
            D_class = D[:, L == i]
            mean_class = D_class.mean(1).reshape(D_class.shape[0], 1)
            cov_class = np.dot((D_class - mean_class), (D_class - mean_class).T) / (D_class.shape[1])

            mean_array.append(mean_class)
            cov_array.append(cov_class)

        self.mean_array = np.array(mean_array)
        self.cov_array = np.array(cov_array)

    def predict_prob(self, D):
        log_S_matrix = np.log(self.__score_matrix__(D, self.prior.shape[0]))
        log_S_Joint = log_S_matrix + np.log(self.prior)
        log_S_marginal = vrow(scipy.special.logsumexp(log_S_Joint, axis=0))
        log_S_post = log_S_Joint - log_S_marginal

        return np.exp(log_S_post)

    def predict(self, D):
        return np.argmax(self.predict_prob(D), 0)
