import os

import numpy as np
import scipy.special
from utils import vcol, vrow

PATH = os.path.abspath(os.path.dirname(__file__))


def confusion_matrix(predictions, labels):
    K = np.unique(labels).size

    cm = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            cm[i, j] = np.sum(np.logical_and(predictions == i, labels == j))

    return cm


def mis_classification_ratio(cm):
    return cm / np.sum(cm, axis=0)


def main():
    # load ll and label for inferno-paradiso
    ll_commedia = np.load(PATH + "/data/commedia_ll.npy")
    ll_commedia_eps1 = np.load(PATH + "/data/commedia_ll_eps1.npy")
    l_commedia = np.load(PATH + "/data/commedia_labels.npy")

    C = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    prior = vcol(np.array([0.3, 0.4, 0.3]))

    # --- Bayes multiclass optimal decision ---

    # --- eps = 0.001 ---

    # compute posterior probability (log joint probability - log marginal densities)
    log_S_Joint = ll_commedia + np.log(prior)
    log_S_marginal = vrow(scipy.special.logsumexp(log_S_Joint, axis=0))
    log_S_post = log_S_Joint - log_S_marginal
    S_post = np.exp(log_S_post)

    C_cap = np.dot(C, S_post)
    predictions = np.argmin(C_cap, axis=0)

    # --- Bayes normalized risk ---
    DCFn = np.min(np.dot(C, prior))

    # --- Bayes risk ---
    cm = confusion_matrix(predictions, l_commedia)
    print(f"Confusion matrix (eps = 0.001):\n{cm}\n")

    # mis-classification ratios
    R = mis_classification_ratio(cm)

    DCFu = np.dot((C * R).sum(0).ravel(), prior.ravel())
    print(f"DCFu (eps = 0.001): {DCFu:.3f}")

    DCF = DCFu / DCFn
    print(f"DCF (eps = 0.001): {DCF:.3f}")
    print("-" * 50)

    # --- eps = 1 ---

    # compute posterior probability (log joint probability - log marginal densities)
    log_S_Joint = ll_commedia_eps1 + np.log(prior)
    log_S_marginal = vrow(scipy.special.logsumexp(log_S_Joint, axis=0))
    log_S_post = log_S_Joint - log_S_marginal
    S_post = np.exp(log_S_post)

    C_cap = np.dot(C, S_post)
    predictions = np.argmin(C_cap, axis=0)

    # --- Bayes normalized risk ---
    DCFn = np.min(np.dot(C, prior))

    # --- Bayes risk ---
    cm = confusion_matrix(predictions, l_commedia)
    print(f"Confusion matrix (eps = 1):\n{cm}\n")

    # mis-classification ratios
    R = mis_classification_ratio(cm)

    DCFu = np.dot((C * R).sum(0).ravel(), prior.ravel())
    print(f"DCFu (eps = 1): {DCFu:.3f}")

    DCF = DCFu / DCFn
    print(f"DCF (eps = 1): {DCF:.3f}")


if __name__ == "__main__":
    main()
