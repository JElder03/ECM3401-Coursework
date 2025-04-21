import numpy as np
import numpy.typing as npt
import random
from sklearn.ensemble import RandomForestClassifier


def symmetric_noise(
    y: list[np.float32], p: float, n_classes: int = None, seed: int = None
) -> npt.NDArray[np.float32]:
    """
    Adds symmetric noise to a set of labels y

    :param y: The label set
    :param p: The probability of mislabelling
    :param n_classes: The number of classes in the dataset. If None, the number is set to the number of
                      unique labels. Note this can be different to the actual number of classes for
                      open-set mislabelling.
    :param seed: An optional random seed
    :return: The labels y with added noise
    """
    if seed:
        np.random.seed(seed)
        random.seed(seed)

    if n_classes is None:
        n_classes = len(set(y))

    for i in range(len(y)):
        if random.random() <= p:
            y[i] += random.randint(1, n_classes - 1) 
            y[i] %= n_classes
    
    return y

def pair_noise(y: list[np.float32], p: float|list[float], n_classes: int = None, unique_pairs : bool = False, seed: int = None) -> npt.NDArray[np.float32]:
    """
    Adds pair noise to a set of labels y with uniform probability of mislabelling

    :param y: The label set
    :param p: A single probability or list of probabilities of mislabelling, one for each label
    :param n_classes: The number of classes in the dataset. If None, the number is set to the number of
                      unique labels. Note this can be different to the actual number of classes for
                      open-set mislabelling.
    :param unique_pairs: Toggle for allowing multiple labels to map to the same label
    :param seed: An optional random seed
    
    :return: The labels y with added noise
    """

    if seed:
        np.random.seed(seed)
        random.seed(seed)

    if n_classes is None:
        n_classes = len(set(y))

    # Create mappings
    pairs = {}
    labels = set(range(n_classes))
    for i in range (n_classes):
        available_labels = list(labels.difference([i]))
        if not available_labels:
            available_labels = list(range(n_classes))
        pairs[i] = random.choice(available_labels)
        
        # Remove mapped to label if unique pairs is enabled
        if unique_pairs:
            labels.discard(pairs[i])

    if type(p) is not list:
        p = [p for _ in range(n_classes)]
    
    # Mislabel
    for i in range(len(y)):
        if random.random() <= p[y[i]]:
            y[i] = pairs[y[i]]
    
    return y

def NNAR(
    X: npt.NDArray[np.float32],
    y: list[int],
    clf: RandomForestClassifier,
    epsilon: float = 1.0,
    seed: int = None
) -> npt.NDArray[np.int_]:
    """
    Adds feature-dependent label noise (NNAR) based on the class posterior outputs
    of a provided scikit-learn Random Forest classifier.

    :param X: The feature matrix (n_samples, n_features)
    :param y: The clean label list (length n_samples)
    :param clf: A trained scikit-learn RandomForestClassifier that outputs posteriors
    :param epsilon: Global noise scaling parameter (higher = more noise)
    :param seed: An optional random seed

    :return: A numpy array of noisy labels
    """
    if seed:
        np.random.seed(seed)
        random.seed(seed)

    X = np.array(X)
    y = np.array(y)
    noisy_y = y.copy()

    # Get posterior probabilities from the RF
    posteriors = clf.predict_proba(X)

    for i in range(len(y)):
        true_label = y[i]
        p_true = posteriors[i][true_label]
        eta_i = min(1.0, epsilon * (1.0 - p_true))

        if random.random() < eta_i:
            # Zero out the true label to avoid sampling it
            probs = posteriors[i].copy()
            probs[true_label] = 0.0
            probs /= probs.sum()  # Renormalize

            # Sample a new label from remaining classes
            new_label = np.random.choice(len(probs), p=probs)
            noisy_y[i] = new_label

    return noisy_y