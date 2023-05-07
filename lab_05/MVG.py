import os

import numpy as np
import scipy
from utils import evaluate_classifier, load_iris, score_matrix, split_db_2to1, vrow

PATH = os.path.abspath(os.path.dirname(__file__))


def mean_cov_estimate(D, L):
    mean_array = []
    cov_array = []

    for i in range(3):
        # select data of each class
        D_class = D[:, L == i]
        # calculate mean of each class
        mean_class = D_class.mean(1).reshape(D_class.shape[0], 1)
        # calculate covariance matrix of each class
        cov_class = np.dot((D_class - mean_class), (D_class - mean_class).T) / (D_class.shape[1])

        mean_array.append(mean_class)
        cov_array.append(cov_class)

    return np.array(mean_array), np.array(cov_array)


def MVG_classifier(D, mean_array, cov_array, prior):
    # compute score matrix for each sample of each class
    S_matrix = score_matrix(D, mean_array, cov_array, prior.shape[0])

    # compute the joint distribution (each row of S_matrix (class-conditional probability) * each prior probability)
    S_joint = S_matrix * prior

    S_marginal = vrow(S_joint.sum(0))

    # compute posterior probability (joint probability / marginal densities)
    S_post = S_matrix / S_marginal

    # joint_sol = np.load(PATH + "/data/SJoint_MVG.npy")
    # print(f"Joint densities error (sol - mine): {np.abs(joint_sol - S_joint).max()}")
    # posterior_sol = np.load(PATH + "/data/Posterior_MVG.npy")
    # print(f"Posterior probability error (sol - mine): {np.abs(posterior_sol - S_post).max()}\n")

    return S_post


def MVG_log_classifier(D, mean_array, cov_array, prior):
    # compute log score matrix for each sample of each class
    log_S_matrix = np.log(score_matrix(D, mean_array, cov_array, prior.shape[0]))

    # compute the log joint distribution (each row of S_matrix (class-conditional probability) * each prior probability)
    log_S_Joint = log_S_matrix + np.log(prior)

    log_S_marginal = vrow(scipy.special.logsumexp(log_S_Joint, axis=0))

    # compute posterior probability (log joint probability - log marginal densities)
    log_S_post = log_S_Joint - log_S_marginal

    # log_S_Joint_sol = np.load(PATH + "/data/logSJoint_MVG.npy")
    # print(f"Log joint density error (sol - mine): {np.abs(log_S_Joint_sol - log_S_Joint).max()}")
    # log_marginal_sol = np.load(PATH + "/data/logMarginal_MVG.npy")
    # print(f"Log marginal density error (sol - mine): {np.abs(log_marginal_sol - log_S_marginal).max()}")
    # log_posterior_sol = np.load(PATH + "/data/logPosterior_MVG.npy")
    # print(f"Log posterior probability error (sol - mine): {np.abs(log_posterior_sol - log_S_post).max()}\n")

    return np.exp(log_S_post)


def main():
    D, L = load_iris()

    # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # --- training ---
    # compute mean and covariance matrix of each class
    mean_array, cov_array = mean_cov_estimate(DTR, LTR)

    # --- classification ---
    prior = np.ones((3, 1)) / 3
    # compute posterior probabilities for samples
    S_post = MVG_classifier(DTE, mean_array, cov_array, prior)

    # compute index of the row where prediction is maximum (each column is a sample)
    predictions = np.argmax(S_post, 0)
    # evaluate gaussian classifier
    accuracy, error_rate = evaluate_classifier(predictions, LTE)

    print(f"MVG model accuracy: {accuracy:.1f}%")
    print(f"MVG model error rate: {error_rate:.1f}%\n")

    # compute posterior probabilities for samples
    S_post = MVG_log_classifier(DTE, mean_array, cov_array, prior)

    predictions = np.argmax(S_post, 0)
    # evaluate log gaussian classifier
    accuracy, error_rate = evaluate_classifier(predictions, LTE)

    print(f"Log MVG model accuracy: {accuracy:.1f}%")
    print(f"Log MVG model error rate: {error_rate:.1f}%\n")


if __name__ == "__main__":
    main()
