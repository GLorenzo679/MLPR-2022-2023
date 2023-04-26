import os
import numpy as np
import sklearn.datasets
import scipy

def vcol(v):
    return v.reshape(v.shape[0], 1)


def vrow(v):
    return v.reshape(1, v.shape[0])


def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

    return D, L


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0/3.0)

    np.random.seed(seed)

    idx = np.random.permutation(D.shape[1])

    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    
    return (DTR, LTR), (DTE, LTE)


def mean_cov_estimate(D, L):
    mean_array = []
    cov_array = []

    for i in range(3):
        # select data of each class
        D_class = D[:, L == i]
        # calculate mean of each class
        mean_class = D_class.mean(1).reshape(D_class.shape[0], 1)
        # calculate covariance matrix of each class
        cov_class = np.dot((D_class-mean_class), (D_class-mean_class).T) / (D_class.shape[1])

        mean_array.append(mean_class)
        cov_array.append(cov_class)

    return np.array(mean_array), np.array(cov_array)


def logpdf_GAU_ND_fast(X, mu, C):
    XC = X - mu
    M = X.shape[0]
    const = -0.5 * M * np.log(2 * np.pi)
    logdet = np.linalg.slogdet(C)[1]
    L = np.linalg.inv(C)
    v = (XC * np.dot(L, XC)).sum(0)

    return const -0.5 * logdet -0.5 * v 


def score_matrix(DTV, mean_array, cov_array):
    S = []

    for i in range(3):
        fcond = np.exp(logpdf_GAU_ND_fast(DTV, mean_array[i], cov_array[i]))
        S.append(vrow(fcond))
    
    return np.vstack(S)


def classifier(D, mean_array, cov_array, prior):
    # compute score matrix for each sample of each class
    S_matrix = score_matrix(D, mean_array, cov_array)

    # compute the joint distribution (each row of S_matrix (class-conditional probability) * each prior probability)
    S_joint = S_matrix * prior

    joint_sol = np.load(os.getcwd() + "/lab_05/data/SJoint_MVG.npy")
    print(f"Joint densities error (sol - mine): {np.abs(joint_sol - S_joint).max()}\n") 
    

    S_marginal = vrow(S_joint.sum(0))
    # compute posterior probability (joint probability / marginal densities)
    S_post = S_matrix / S_marginal
    posterior_sol = np.load(os.getcwd() + "/lab_05/data/Posterior_MVG.npy")
    print(f"Posterior probability error (sol - mine): {np.abs(posterior_sol - S_post).max()}\n") 

    return S_post

def log_classifier(D, mean_array, cov_array, prior):
    # compute log score matrix for each sample of each class
    log_S_matrix = np.log(score_matrix(D, mean_array, cov_array))

    # compute the log joint distribution (each row of S_matrix (class-conditional probability) * each prior probability)
    log_S_Joint = log_S_matrix * prior
    log_S_Joint_sol = np.load(os.getcwd() + "/lab_05/data/logSJoint_MVG.npy")
    print(f"Log joint density error (sol - mine): {np.abs(log_S_Joint_sol - log_S_Joint).max()}\n") 

    #joint_sol = np.load(os.getcwd() + "/lab_05/data/SJoint_MVG.npy")
    #print(f"Joint densities error (mine - sol): {np.abs(log_S_Joint - joint_sol).max()}\n") 
    

    log_S_marginal = vrow(scipy.special.logsumexp(log_S_Joint, axis=0))
    log_marginal_sol = np.load(os.getcwd() + "/lab_05/data/logPosterior_MVG.npy")
    print(f"Log arginal density error (sol - mine): {np.abs(log_marginal_sol - log_S_marginal).max()}\n") 

    # compute posterior probability (log joint probability - log marginal densities)
    log_S_post = log_S_Joint - log_S_marginal
    log_posterior_sol = np.load(os.getcwd() + "/lab_05/data/logMarginal_MVG.npy")
    print(f"Log posterior probability error (sol - mine): {np.abs(log_posterior_sol - log_S_post).max()}\n") 

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
    mean_array, cov_array = mean_cov_estimate(DTR, LTR)

    # --- classification ---
    prior = np.ones((3,1)) / 3
    # compute posterior probabilities for samples
    S_post = classifier(DTE, mean_array, cov_array, prior)

    # compute index of the row where prediction is maximum (each column is a sample)
    predictions = np.argmax(S_post, 0)
    # evaluate gaussian classifier
    accuracy, error_rate = evaluate_classifier(predictions, LTE)

    print(f"Gaussian model accuracy: {accuracy:.2f}\n")
    print(f"Gaussian model error rate: {error_rate:.2f}\n")

    # compute posterior probabilities for samples
    S_post = log_classifier(DTE, mean_array, cov_array, prior)

    predictions = np.argmax(S_post, 0)
    # evaluate log gaussian classifier
    accuracy, error_rate = evaluate_classifier(predictions, LTE)

    print(f"Log gaussian model accuracy: {accuracy:.2f}\n")
    print(f"Log gaussian model error rate: {error_rate:.2f}\n")


if __name__ == "__main__":
    main()