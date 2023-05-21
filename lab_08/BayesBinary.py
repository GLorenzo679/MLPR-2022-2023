import os

import matplotlib.pyplot as plt
import numpy as np

PATH = os.path.abspath(os.path.dirname(__file__))


def confusion_matrix(predictions, labels):
    K = np.unique(labels).size

    cm = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            cm[i, j] = np.sum(np.logical_and(predictions == i, labels == j))

    return cm


def optimal_bayes_decision_binary(llr, p, Cfn, Cfp):
    threshold = -np.log((p * Cfn) / ((1 - p) * Cfp))
    predictions = llr > threshold

    return predictions.astype(int)


def bayes_risk(cm, p, Cfn, Cfp):
    FNR = cm[0, 1] / (cm[0, 1] + cm[1, 1])
    FPR = cm[1, 0] / (cm[0, 0] + cm[1, 0])

    DCFu = p * Cfn * FNR + (1 - p) * Cfp * FPR

    return DCFu


def normalized_bayes_risk(cm, p, Cfn, Cfp):
    FNR = cm[0, 1] / (cm[0, 1] + cm[1, 1])
    FPR = cm[1, 0] / (cm[0, 0] + cm[1, 0])

    C = min(p * Cfn, (1 - p) * Cfp)

    DCFd = (p * Cfn * FNR + (1 - p) * Cfp * FPR) / C

    return DCFd


def minimum_bayes_risk(llr, l_infpar, p, thresholds, Cfn, Cfp):
    min_t = []

    for t in thresholds:
        predictions = llr > t
        cm = confusion_matrix(predictions.astype(int), l_infpar)
        min_t.append(normalized_bayes_risk(cm, p, Cfn, Cfp))

    return min(min_t)


def ROC(llr, l_infpar, thresholds):
    FPR = []
    TPR = []

    for t in thresholds:
        predictions = llr > t
        cm = confusion_matrix(predictions.astype(int), l_infpar)
        FPR.append(cm[1, 0] / (cm[0, 0] + cm[1, 0]))
        TPR.append(1 - (cm[0, 1] / (cm[0, 1] + cm[1, 1])))

    plt.plot(FPR, TPR)
    plt.grid(visible=True, linestyle="dotted")
    plt.ylabel("TPR")
    plt.xlabel("FPR")
    plt.show()


def main():
    # load llr and label for inferno-paradiso
    llr_infpar = np.load(PATH + "/data/commedia_llr_infpar.npy")
    llr_infpar_eps1 = np.load(PATH + "/data/commedia_llr_infpar_eps1.npy")
    l_infpar = np.load(PATH + "/data/commedia_labels_infpar.npy")

    # --- Bayes binary optimal decision ---
    cost_matrices = [
        np.array([[0, 1], [1, 0]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, 10], [1, 0]]),
        np.array([[0, 1], [10, 0]]),
        np.array([[0, 10], [10, 0]]),
    ]

    priors = [0.5, 0.8, 0.5, 0.8]

    for C, p in zip(cost_matrices, priors):
        Cfn = C[0, 1]
        Cfp = C[1, 0]
        predictions = optimal_bayes_decision_binary(llr_infpar, p, Cfn, Cfp)
        cm = confusion_matrix(predictions, l_infpar)
        print(f"Parameters:\n-Prior:\n{p}\n-Cost matrix:\n{C}\n\nConfusion matrix:\n{cm}\n")

        # --- Bayes risk ---
        DCFu = bayes_risk(cm, p, Cfn, Cfp)
        print(f"DCF: {DCFu:.3f}\n")

        # --- Bayes normalized risk ---
        DCFd = normalized_bayes_risk(cm, p, Cfn, Cfp)
        print(f"DCF dummy: {DCFd:.3f}\n")

        # --- Bayes minimum risk ---
        thresholds = np.sort(llr_infpar)
        min_bayes_risk = minimum_bayes_risk(llr_infpar, l_infpar, p, thresholds, Cfn, Cfp)
        print(f"Minimum Bayes risk: {min_bayes_risk:.3f}\n")

        # --- ROC ---
        ROC(llr_infpar, l_infpar, thresholds)

        print("-" * 50)

    # --- Bayes error plots ---

    eff_prior_log_odds = np.linspace(-3, 3, 21)
    eff_prior = 1 / (1 + np.exp(-eff_prior_log_odds))

    DCF = []
    minDCF = []
    DCF_eps1 = []
    minDCF_eps1 = []
    Cfn = 1
    Cfp = 1

    for ep in eff_prior:
        predictions = optimal_bayes_decision_binary(llr_infpar, ep, Cfn, Cfp)
        cm = confusion_matrix(predictions, l_infpar)
        DCF.append(normalized_bayes_risk(cm, ep, Cfn, Cfp))
        minDCF.append(minimum_bayes_risk(llr_infpar, l_infpar, ep, thresholds, Cfn, Cfp))

        predictions_eps1 = optimal_bayes_decision_binary(llr_infpar_eps1, ep, Cfn, Cfp)
        cm_eps1 = confusion_matrix(predictions_eps1, l_infpar)
        DCF_eps1.append(normalized_bayes_risk(cm_eps1, ep, Cfn, Cfp))
        minDCF_eps1.append(minimum_bayes_risk(llr_infpar_eps1, l_infpar, ep, thresholds, Cfn, Cfp))

    plt.plot(eff_prior_log_odds, DCF, label="DCF (ε = 0.001)", color="r", linewidth=2)
    plt.plot(eff_prior_log_odds, minDCF, label="min DCF (ε = 0.001)", color="b", linewidth=2)
    plt.plot(eff_prior_log_odds, DCF_eps1, label="DCF (ε = 1)", color="gold", linewidth=2)
    plt.plot(eff_prior_log_odds, minDCF_eps1, label="min DCF (ε = 1)", color="c", linewidth=2)
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.ylabel("DCF value")
    plt.xlabel("prior log-odds")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
