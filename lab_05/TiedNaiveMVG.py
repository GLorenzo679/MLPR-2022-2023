import os

import numpy as np
import scipy
from utils import (
    evaluate_classifier,
    load_iris,
    logpdf_GAU_ND_fast,
    split_db_2to1,
    vrow,
)

PATH = os.path.abspath(os.path.dirname(__file__))


def mean_cov_estimate(D, L):
    mean_array = []
    class_cov = 0

    for i in range(3):
        # select data of each class
        D_class = D[:, L == i]
        # calculate mean of each class
        mean_class = D_class.mean(1).reshape(D_class.shape[0], 1)
        # calculate covariance matrix of each class
        class_cov += np.dot((D_class - mean_class), (D_class - mean_class).T)

        mean_array.append(mean_class)

    # compute within class covariance
    within_class_cov = class_cov / (D.shape[1])
    # calculate diagonal of within class covariance matrix
    diag_within_class_cov = within_class_cov * np.identity(within_class_cov.shape[0])

    return np.array(mean_array), diag_within_class_cov


def score_matrix(DTV, mean_array, within_class_cov):
    S = []

    for i in range(3):
        fcond = np.exp(logpdf_GAU_ND_fast(DTV, mean_array[i], within_class_cov))
        S.append(vrow(fcond))

    return np.vstack(S)


def tied_naive_MVG_classifier(D, mean_array, within_class_cov, prior):
    # compute score matrix for each sample of each class
    S_matrix = score_matrix(D, mean_array, within_class_cov)

    # compute the joint distribution (each row of S_matrix (class-conditional probability) * each prior probability)
    S_joint = S_matrix * prior

    S_marginal = vrow(S_joint.sum(0))

    # compute posterior probability (joint probability / marginal densities)
    S_post = S_matrix / S_marginal

    # joint_sol = np.load(PATH + "/data/SJoint_TiedNaiveBayes.npy")
    # print(f"Joint densities error (sol - mine): {np.abs(joint_sol - S_joint).max()}")
    # posterior_sol = np.load(PATH + "/data/Posterior_TiedNaiveBayes.npy")
    # print(f"Posterior probability error (sol - mine): {np.abs(posterior_sol - S_post).max()}\n")

    return S_post


def tied_naive_MVG_log_classifier(D, mean_array, within_class_cov, prior):
    # compute log score matrix for each sample of each class
    log_S_matrix = np.log(score_matrix(D, mean_array, within_class_cov))

    # compute the log joint distribution (each row of S_matrix (class-conditional probability) * each prior probability)
    log_S_Joint = log_S_matrix + np.log(prior)

    log_S_marginal = vrow(scipy.special.logsumexp(log_S_Joint, axis=0))

    # compute posterior probability (log joint probability - log marginal densities)
    log_S_post = log_S_Joint - log_S_marginal

    # log_S_Joint_sol = np.load(PATH + "/data/logSJoint_TiedNaiveBayes.npy")
    # print(f"Log joint density error (sol - mine): {np.abs(log_S_Joint_sol - log_S_Joint).max()}")
    # log_marginal_sol = np.load(PATH + "/data/logMarginal_TiedNaiveBayes.npy")
    # print(f"Log marginal density error (sol - mine): {np.abs(log_marginal_sol - log_S_marginal).max()}")
    # log_posterior_sol = np.load(PATH + "/data/logPosterior_TiedNaiveBayes.npy")
    # print(f"Log posterior probability error (sol - mine): {np.abs(log_posterior_sol - log_S_post).max()}\n")

    return np.exp(log_S_post)


def evaluate_classifier(predictions, labels):
    # compute boolean array, true if prediction == eval label else false
    matched = np.array([True if x1 == x2 else False for x1, x2 in zip(predictions, labels)])

    # sum totale number of True (correct predictions) and divide by number of samples
    accuracy = matched.sum() / predictions.size
    error_rate = 1 - accuracy

    return accuracy, error_rate


def main():
    D, L = load_iris()

    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # --- training ---
    # compute mean and covariance matrix of each class
    mean_array, within_class_cov = mean_cov_estimate(DTR, LTR)

    # --- classification ---
    prior = np.ones((3, 1)) / 3
    # compute posterior probabilities for samples
    S_post = tied_naive_MVG_classifier(DTE, mean_array, within_class_cov, prior)

    # compute index of the row where prediction is maximum (each column is a sample)
    predictions = np.argmax(S_post, 0)
    # evaluate gaussian classifier
    accuracy, error_rate = evaluate_classifier(predictions, LTE)

    print(f"Tied naive MVG model accuracy: {accuracy:.2f}%")
    print(f"Tied naive MVG model error rate: {error_rate:.2f}%\n")

    # compute posterior probabilities for samples
    S_post = tied_naive_MVG_log_classifier(DTE, mean_array, within_class_cov, prior)

    predictions = np.argmax(S_post, 0)
    # evaluate log gaussian classifier
    accuracy, error_rate = evaluate_classifier(predictions, LTE)

    print(f"Log tied naive MVG model accuracy: {accuracy:.2f}%")
    print(f"Log tied naive MVG model error rate: {error_rate:.2f}%\n")


if __name__ == "__main__":
    main()
