import os

import numpy as np
from utils import MVGLogClassifier, load_iris, split_db_2to1

PATH = os.path.abspath(os.path.dirname(__file__))


def confusion_matrix(predictions, labels):
    K = np.unique(labels).size

    cm = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            cm[i, j] = np.sum(np.logical_and(predictions == i, labels == j))

    return cm


def main():
    D, L = load_iris()

    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    prior = np.ones((3, 1)) / 3
    mvg_log_classifier = MVGLogClassifier(prior)

    # --- training ---
    # compute mean and covariance matrix of each class
    mvg_log_classifier.train(DTR, LTR)

    # --- classification ---
    # compute index of the row where prediction is maximum (each column is a sample)
    predictions = mvg_log_classifier.predict(DTE)

    # compute confusion matrix for iris dateset
    cm = confusion_matrix(predictions, LTE)

    print(f"Confusion matrix for IRIS dataset:\n\n{cm}\n")

    likelihood = np.load(PATH + "/data/commedia_ll.npy")
    LTE = np.load(PATH + "/data/commedia_labels.npy")
    predictions = np.argmax(likelihood, axis=0)

    # compute confusion matrix for commedia dateset
    cm = confusion_matrix(predictions, LTE)

    print(f"Confusion matrix for Commedia dataset:\n\n{cm}\n")


if __name__ == "__main__":
    main()
