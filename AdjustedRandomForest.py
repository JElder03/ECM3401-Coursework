import math
from collections import defaultdict
import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.utils import shuffle


def train(
    ensemble: RandomForestClassifier | ExtraTreesClassifier,
    X: list,
    y: list,
    labels: list,
    n_estimators: int = 5,
    iterations=10,
    K: float = 0.5,
    L: float = 0.01,
    B: float = 0.02,
    initial_certainty: float = 0.95,
    bootstrapping: bool = False,
    relabelling: bool = True,
) -> tuple[RandomForestClassifier | ExtraTreesClassifier, list]:
    """
    Trains an ensemble classifier using a custom reweighting approach and simultaneously relabelled noisy samples.

    :param ensemble: Type of ensemble classifier (RandomForest or ExtraTrees)
    :param X: Feature set
    :param y: Labels
    :param labels: List of possible class labels
    :param n_estimators: Number of weak classifiers to train
    :param iterations: Number of iterative training steps
    :param K: Selection threshold
    :param L: Learning rate for updating class labels
    :param B: Parameter for sigmoid scaling
    :param initial_certainty: Initial certainty of labels
    :param bootstrapping: Enables the bootstrap-based training strategy
    :param relabelling: Enables the relabelling capability

    :return: Trained ensemble model
    :return: Updated label predictions
    """

    NUM_CLASSES = len(labels)
    batch_size = math.floor(len(y) / n_estimators)
    y_ordered = np.copy(y)
    X_ordered = np.copy(X)

    # Create probability matrix for sample labels
    y = np.array(y)
    p = np.eye(NUM_CLASSES)[y]
    p[p == 0] = (1 - initial_certainty) / (NUM_CLASSES - 1)
    p[p == 1] = initial_certainty

    u = np.array(
        [
            [inv_sigmoid(probability, B) for probability in probabilities]
            for probabilities in p
        ]
    )

    forest = None

    # Iteratively generate ensembles using knowledge learnt from the previous ensemble
    for _ in range(iterations):
        forest_prev = forest

        # Reinitialize new ensemble with depth limitation
        forest = ensemble(
            n_estimators=1, criterion="entropy", bootstrap=bootstrapping, warm_start= not bootstrapping
        )

        if bootstrapping:
            if forest_prev is not None:
                e = forest_prev.predict_proba(X)
                selection = select_training_data(X, y, e, K)
                X_train = [X[i] for i in selection]
                y_train = [y[i] for i in selection]
            else:
                X_train = X
                y_train = y

            missing_labels = set(labels) - set(y_train)
            for label in missing_labels:
                y_train = np.append(y_train, label)
                if np.array(X_train).size == 0:  # Ensure it's properly initialized
                    X_train = np.zeros((1, np.array(X).shape[1]))  
                else:
                    X_train = np.vstack((X_train, np.zeros((1, np.array(X).shape[1]))))

            forest.set_params(n_estimators=n_estimators)
            forest.fit(X_train, y_train)

        else:
            X, y = shuffle(X, y)
            for i in range(n_estimators):
                X_batch = X[i * batch_size : (i + 1) * batch_size]
                y_batch = y[i * batch_size : (i + 1) * batch_size]

                if forest_prev is not None:
                    # Get predicted probabilities from previous iteration
                    e = forest_prev.predict_proba(X_batch)

                    # Select training data based on probability confidence
                    selection = select_training_data(X_batch, y_batch, e, K)
                    X_batch = [X_batch[i] for i in selection]
                    y_batch = [y_batch[i] for i in selection]

                # Ensure all labels exist in the batch
                missing_labels = set(labels) - set(y_batch)
                for label in missing_labels:
                    y_batch = np.append(y_batch, label)
                    if np.array(X_batch).size == 0:  # Ensure it's properly initialized
                        X_batch = np.zeros((1, np.array(X).shape[1]))
                    else:
                        X_batch = np.vstack(
                            (X_batch, np.zeros((1, np.array(X).shape[1])))
                        )

                # Ensure added samples have 0 weight
                class_weights = {label: 1 for label in set(labels)} | {
                    label: 0 for label in missing_labels
                }

                forest.set_params(n_estimators=i + 1, class_weight=class_weights)
                forest.fit(X_batch, y_batch)

        if relabelling:
            # Update confidence scores, predictions, and labels
            e = forest.predict_proba(X_ordered)
            u = update_us(u, e, p, L)
            p = update_probabilities(u, B)
            y = update_labels(y_ordered, p)
            X = np.copy(X_ordered)

    return forest, y


def select_training_data(
    X: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
    e: npt.NDArray[np.float32],
    K: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Selects training samples with high confidence scores.

    :param X: Feature set
    :param y: Labels
    :param e: Sample prediction probability matrix
    :param K: Selection threshold
    """

    # group samples by label
    grouped_values = defaultdict(list)

    for label, probability in zip(y, e):
        grouped_values[label].append(probability[label])

    # calculate mean and std of label certainty for each label
    means = {
        label: np.mean(probabilities) for label, probabilities in grouped_values.items()
    }
    standard_deviations = {
        label: np.std(probabilities) for label, probabilities in grouped_values.items()
    }

    # select samples with sufficiently certain labels
    selection = []
    for i in range(len(X)):
        if e[i][y[i]] >= (means[y[i]] + K * standard_deviations[y[i]]):
            selection.append(i)

    return np.array(selection)


def update_us(
    u: npt.NDArray[np.float32],
    e: npt.NDArray[np.float32],
    p: npt.NDArray[np.float32],
    L: float,
) -> npt.NDArray[np.float32]:
    """
    Updates pre-sigmoid confidence values
    """

    return u + L * (e - p)


def update_probabilities(u: npt.NDArray[np.float32], B: float):
    """
    Updates label probability values using a sigmoid transformation.
    """

    p = np.array(
        [
            [sigmoid(probability, B) for probability in probabilities]
            for probabilities in u
        ]
    )  # calculate p from u (sigmoid)

    # Normalize values
    for i in range(len(p)):
        total = np.sum(p[i])
        p[i] = p[i] / total
    return p


def update_labels(
    y: npt.NDArray[np.float32], p: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Updates labels based on maximum probability.
    """

    for i in range(len(y)):
        y[i] = np.argmax(p[i])
    return y


def sigmoid(x, b):
    """
    Sigmoid function with scaling parameter.
    """

    return (1 + math.tanh(x / b)) / 2


def inv_sigmoid(x, b):
    """
    Inverse sigmoid function with scaling parameter.
    """

    x = max(0.0001, min(0.9999, x))  # Clamp x to avoid out-of-domain errors
    return math.atanh(2 * x - 1) * b
