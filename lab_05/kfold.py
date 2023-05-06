import MVG
import NaiveMVG as NMVG
import numpy as np
import TiedMVG as TMVG
import TiedNaiveMVG as TNMVG
from utils import evaluate_classifier, load_iris


def k_fold_cross_validation(D, L, k, seed=0):
    data_partitions = []

    np.random.seed(seed)

    # idx = np.random.permutation(D.shape[1])
    idx = np.arange(D.shape[1])

    # n_samples / k = size of single partition
    size_partition = int(D.shape[1] / k)

    for i in range(k):
        idx_start_test = i * size_partition
        idx_end_test = (i + 1) * size_partition

        idxTest = idx[idx_start_test:idx_end_test]
        idxTrain = idx[0:idx_start_test]
        idxTrain = np.append(idxTrain, (idx[idx_end_test:]))

        # DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]

        data_partitions.append(((DTR, LTR), (DTE, LTE)))

    return data_partitions


def main():
    D, L = load_iris()

    data_partitions = k_fold_cross_validation(D, L, D.shape[1])

    prior = np.ones((3, 1)) / 3

    classifiers = {
        "MVG log classifier": [MVG.MVG_log_classifier, MVG.mean_cov_estimate],
        "Naive MVG log classifier": [NMVG.naive_MVG_log_classifier, NMVG.mean_cov_estimate],
        "Tied MVG log classifier": [TMVG.tied_MVG_log_classifier, TMVG.mean_cov_estimate],
        "Tied naive MVG log classifier": [TNMVG.tied_naive_MVG_log_classifier, TNMVG.mean_cov_estimate],
    }

    for name, classifier in classifiers.items():
        tot_error = 0

        for dp in data_partitions:
            (DTR, LTR), (DTE, LTE) = dp

            # --- training ---
            mean_array, cov_array = classifier[1](DTR, LTR)

            # --- classification ---
            S_post = classifier[0](DTE, mean_array, cov_array, prior)
            predictions = np.argmax(S_post, 0)
            _, error_rate = evaluate_classifier(predictions, LTE)

            tot_error += error_rate

        tot_error_rate = tot_error / D.shape[1]

        print(f'Error rate for "{name}": {tot_error_rate:.1f}%')


if __name__ == "__main__":
    main()
