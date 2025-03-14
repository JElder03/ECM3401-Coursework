import numpy as np
import numpy.typing as npt
import random


def symmetric_noise(
    y: list[np.float32], p: float, n_classes: int = None
) -> npt.NDArray[np.float32]:
    """
    Adds symmetric noise to a set of labels y

    :param y: The label set
    :param p: The probability of mislabelling
    :param n_classes: The number of classes in the dataset. If None, the number is set to the number of
                      unique labels. Note this can be different to the actual number of classes for
                      open-set mislabelling.
    :return: The labels y with added noise
    """

    if n_classes is None:
        n_classes = len(set(y))

    for i in range(len(y)):
        if random.random() <= p:
            y[i] += random.randint(1, n_classes - 1) 
            y[i] %= n_classes
    
    return y

def pair_noise(y: list[np.float32], p: float|list[float], n_classes: int = None, unique_pairs : bool = False) -> npt.NDArray[np.float32]:
    """
    Adds pair noise to a set of labels y with uniform probability of mislabelling

    :param y: The label set
    :param p: A single probability or list of probabilities of mislabelling, one for each label
    :param n_classes: The number of classes in the dataset. If None, the number is set to the number of
                      unique labels. Note this can be different to the actual number of classes for
                      open-set mislabelling.
    :param unique_pairs: Toggle for allowing multiple labels to map to the same label
    :return: The labels y with added noise
    """

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