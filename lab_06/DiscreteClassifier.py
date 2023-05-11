import numpy as np
import scipy
from load import load_data, split_data


def vrow(v):
    return v.reshape(1, v.shape[0])


def mcol(m):
    return m.reshape((m.size, 1))


def words_to_dict(l_tercets, eps):
    """
    Computes a dictionary where each key is a word and the values are initialized to eps.

    Parameters:
    ---
    l_tercets: list of all the tercets
    eps: smoothing factor (pseudo-count)

    Returns:
    ---
    dict_words: dictionary with a key for every word and eps as a value
    """

    dict_words = dict()

    for line in l_tercets:
        for word in line.split():
            if word not in dict_words.keys():
                dict_words[word] = eps

    return dict_words


def norm_freq_words(dict_words, l_cls):
    """
    Computes a list of dictionaries. Each element is a dictionary where each key is a word and the value is the corresponding normalized log frequency.

    Parameters:
    ---
    dict_words: dictionary with a key for every word and eps as a value
    l_cls: list of the 3 classes

    Returns:
    ---
    l_cls_freq: a list of dictionaries (one for each cantica) with word-freq key value pairs.
    """

    l_cls_freq = []

    for cls in l_cls:
        cls_freq = dict_words.copy()
        n_cls_elem = 0

        for line in cls:
            for word in line.split():
                # increment count for single word in class, and increment total word count for the class
                cls_freq[word] += 1
                n_cls_elem += 1

        x = sum(cls_freq.values())

        for k in cls_freq.keys():
            cls_freq[k] = np.log(cls_freq[k]) - np.log(x)

        l_cls_freq.append(cls_freq)

    return l_cls_freq


def loglikelihood_array(l_cls_freq, text):
    """
    Computes an array of class conditional log likelihood.

    Parameters:
    ---
    l_cls_freq: a list of dictionaries (one for each cantica) with word-freq key value pairs.
    text: text (in this case a tercets) to compute ll from.

    Returns:
    ---
    l_ll: list of class conditional log likelihood for text (tercets)
    """

    l_ll = []

    for cls in l_cls_freq:
        ll = 0

        for word in text.split():
            if word in cls.keys():
                ll += cls[word]

        l_ll.append(ll)

    return l_ll


def loglikelihood_matrix(l_cls_freq, l_tercets):
    """
    Compute the log likelihood matrix.

    Parameters:
    ---
    l_cls_freq: list of classes, each class is a dictionary with word-freq key-value pairs
    l_tercets: list of all tercets for which we have to compute log ll

    Return:
    ---
    S: matrix of log likelihood (rows -> class, columns -> log ll for each tercet)
    """
    S = []

    for tercet in l_tercets:
        S.append(np.array(loglikelihood_array(l_cls_freq, tercet)))

    return np.array(S).T


def compute_class_posterior(S, prior):
    S_joint = S + mcol(prior)

    S_marginal = vrow(scipy.special.logsumexp(S_joint, axis=0))

    S_post = S_joint - S_marginal

    return np.exp(S_post)


def evaluate(predictions, L):
    matched = np.array([True if x1 == x2 else False for x1, x2 in zip(predictions, L)])

    accuracy = matched.sum() / predictions.size

    return 100 * accuracy


def main():
    # load dataset
    lInf, lPur, lPar = load_data()

    # split data in train set and evaluation set
    lInf_train, lInf_evaluation = split_data(lInf, 4)
    lPur_train, lPur_evaluation = split_data(lPur, 4)
    lPar_train, lPar_evaluation = split_data(lPar, 4)

    # create a dictionary of all the words in the training set
    dict_words = words_to_dict(lInf_train + lPur_train + lPar_train, eps=0.001)

    # create a list of dictionary with word-freq per class
    l_cls_freq = norm_freq_words(dict_words, [lInf_train, lPur_train, lPar_train])

    # compute likelihood matrix for evaluation set
    S = loglikelihood_matrix(l_cls_freq, lInf_evaluation + lPur_evaluation + lPar_evaluation)

    # ----- Multiclass -----

    prior = np.log(np.ones((3, 1)) / 3)

    S_post = compute_class_posterior(S, prior)

    # check best prediction in each column (axis = 0)
    predictions = np.argmax(S_post, axis=0)

    # create class labels
    label_inf = np.ones(len(lInf_evaluation)) * 0
    label_pur = np.ones(len(lPur_evaluation)) * 1
    label_par = np.ones(len(lPar_evaluation)) * 2

    L = np.hstack([label_inf, label_pur, label_par])

    # create indeces for label array
    idx_label = [label_inf.size, label_inf.size + label_pur.size, label_inf.size + label_pur.size + label_par.size]

    # evaluate predictions made by classifier
    inf_accuracy = evaluate(predictions[: idx_label[0]], label_inf)
    pur_accuracy = evaluate(predictions[idx_label[0] : idx_label[1]], label_pur)
    par_accuracy = evaluate(predictions[idx_label[1] : idx_label[2]], label_par)
    tot_accuracy = evaluate(predictions, L)

    print(f"Multiclass - Inferno accuracy: {round(inf_accuracy)}%")
    print(f"Multiclass - Purgatorio accuracy: {round(pur_accuracy)}%")
    print(f"Multiclass - Paradiso accuracy: {round(par_accuracy)}%")
    print(f"Multiclass - Total accuracy: {round(tot_accuracy)}%")

    # ----- Binary -----

    prior = np.log(np.ones((2, 1)) / 2)
    Sb = np.hstack((S[0:3:2, :400], S[0:3:2, 802:]))
    # Sb = S[0:3:2, :]
    S_post = compute_class_posterior(Sb, prior)
    predictions = np.argmax(S_post, axis=0)
    label_inf = np.ones(len(lInf_evaluation)) * 0
    label_par = np.ones(len(lPar_evaluation)) * 1
    L = np.hstack([label_inf, label_par])
    tot_accuracy = evaluate(predictions, L)
    print(f"\nBinary - Inferno/Paradiso accuracy: {round(tot_accuracy)}%")

    prior = np.log(np.ones((2, 1)) / 2)
    Sb = S[0:2, 0:802]
    S_post = compute_class_posterior(Sb, prior)
    predictions = np.argmax(S_post, axis=0)
    label_inf = np.ones(len(lInf_evaluation)) * 0
    label_pur = np.ones(len(lPur_evaluation)) * 1
    L = np.hstack([label_inf, label_pur])
    tot_accuracy = evaluate(predictions, L)

    print(f"Binary - Inferno/Purgatorio accuracy: {round(tot_accuracy)}%")

    prior = np.log(np.ones((2, 1)) / 2)
    Sb = S[1:3, 400:]
    S_post = compute_class_posterior(Sb, prior)
    predictions = np.argmax(S_post, axis=0)
    label_pur = np.ones(len(lInf_evaluation)) * 0
    label_par = np.ones(len(lPar_evaluation)) * 1
    L = np.hstack([label_pur, label_par])
    tot_accuracy = evaluate(predictions, L)

    print(f"Binary - Purgatorio/Paradiso accuracy: {round(tot_accuracy)}%")


if __name__ == "__main__":
    main()
