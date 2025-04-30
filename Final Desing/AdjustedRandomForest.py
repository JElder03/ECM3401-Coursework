import math
from collections import defaultdict
import numpy as np
import numpy.typing as npt
from typing import Union
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.utils import shuffle


def train(
    ensemble: Union[type[RandomForestClassifier], type[ExtraTreesClassifier]],
    X: npt.NDArray[np.float32],
    y: npt.NDArray[np.int32],
    n_estimators: int = 5,
    iterations: int = 10,
    K: float = 0.5,
    L: float = 0.01,
    B: float = 0.02,
    initial_certainty: float = 0.95,
    bootstrapping: bool = False,
    relabelling: bool = True,
    labels: npt.NDArray[np.int32] = None,
    max_features = 'sqrt'
) -> tuple[RandomForestClassifier | ExtraTreesClassifier, npt.NDArray[np.int32]]:
    """
    Trains an ensemble classifier using iterative reweighting and optional relabelling.

    Args:
        ensemble: The ensemble classifier class (RandomForestClassifier or ExtraTreesClassifier).
        X: Training feature matrix of shape (n_samples, n_features).
        y: Initial hard labels for each sample (length n_samples).
        n_estimators: Number of estimators to train per iteration.
        iterations: Number of total training iterations.
        K: Confidence threshold factor for selecting training samples.
        L: Learning rate for updating soft label logits.
        B: Sigmoid scaling parameter.
        initial_certainty: Initial confidence assigned to known labels.
        bootstrapping: If True, use bootstrap sampling in the final iteration.
        relabelling: If True, update soft and hard labels during training.

    Returns:
        A trained ensemble model and the final relabelled hard labels.
    """

    if labels is None:
        labels = sorted(np.unique(y))
    NUM_CLASSES = len(labels)
    batch_size = math.floor(len(y) / n_estimators)

    # Store original data order for consistent relabelling
    y_ordered = np.copy(y)
    X_ordered = np.copy(X)

    # Create soft label matrix from hard labels with initial certainty
    p_ordered = np.eye(NUM_CLASSES)[y]
    p_ordered[p_ordered == 0] = (1 - initial_certainty) / (NUM_CLASSES - 1)
    p_ordered[p_ordered == 1] = initial_certainty

    # Convert to logits
    u = np.vectorize(inv_sigmoid)(p_ordered, B)

    # Instantiate ensemble
    forest = ensemble(criterion="entropy", bootstrap=False, warm_start=True, max_features = max_features)
    fitted = False

    for iter in range(iterations):
        if iter == (iterations - 1):
            bootstrapping = True  # Enable bootstrapping for final training

        # Shuffle current data and probabilities
        X, y, p = shuffle(X_ordered, y_ordered, p_ordered)

        # Predict with current ensemble
        if fitted:
            e = slice_prediction(X, forest, n_estimators)
        else:
            e = np.copy(p)

        for i in range(n_estimators):  
            # Select training indices (bootstrap or chunk)
            if bootstrapping:
                indices = np.random.choice(len(X), size=len(X), replace=True)
            else:
                indices = list(range(i * batch_size, (i + 1) * batch_size))

            # Confidence-based sample selection
            selection = select_training_data(X, y, e, K)
            if len(set(selection) & set(indices)) != 0:
                indices = [i for i in indices if i in selection]

            # Prepare training data
            X_train = X[indices]
            y_train = y[indices]

            # Ensure all classes are represented with dummy samples (0 weight)
            missing_labels = set(labels) - set(y_train)
            if missing_labels:
                dummy_X = np.zeros((len(missing_labels), X.shape[1]), dtype=np.float32)
                dummy_y = np.array(list(missing_labels), dtype=np.int32)
                X_train = np.vstack([X_train, dummy_X])
                y_train = np.concatenate([y_train, dummy_y])
                sample_weight = np.ones(len(X_train))
                sample_weight[-len(missing_labels):] = 0  # Set dummy weights to 0
            else:
                sample_weight = np.ones(len(X_train))

            # Update ensemble with new estimator
            forest.set_params(n_estimators=n_estimators * iter + i + 1)
            forest.fit(X_train, y_train, sample_weight=sample_weight)

        fitted = True

        # Optional relabelling step: update soft and hard labels
        if relabelling:
            e = slice_prediction(X_ordered, forest, n_estimators)
            u = update_us(u, e, p_ordered, L)
            p_ordered = update_probabilities(u, B)
            y_ordered = update_labels(p_ordered)

    return forest, y_ordered

def slice_prediction(
    X: npt.NDArray[np.float32],
    forest: Union[RandomForestClassifier, ExtraTreesClassifier],
    t: int
) -> npt.NDArray[np.float32]:
    '''
    Computes the average predicted class probabilities across the last `t` trees in the ensemble.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        forest: Trained RandomForestClassifier or ExtraTreesClassifier.
        t: Number of most recent estimators to average over.

    Returns:
        A (n_samples, n_classes) array of averaged class probabilities.
    '''

    e = np.zeros((len(X), len(forest.classes_)))
    for tree in forest.estimators_[-t:]:
        e += tree.predict_proba(X)
    return e / t

def select_training_data(
    X: npt.NDArray[np.float32],
    y: npt.NDArray[np.int32],
    e: npt.NDArray[np.float32],
    K: float,
) -> npt.NDArray[np.int32]:
    """
    Selects training samples with high confidence scores.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Ground-truth/pseudo labels for each sample, shape (n_samples,).
        e: Ensemble prediction probabilities of shape (n_samples, n_classes).
        K: Confidence threshold multiplier.

    Returns:
        A 1D array of selected sample indices with high-confidence predictions.
    """

    # Collect per-class predicted probabilities for their own label
    grouped_probs = defaultdict(list)
    for i in range(len(y)):
        grouped_probs[y[i]].append(e[i][y[i]])

    # Compute class-specific thresholds
    means = {label: np.mean(probs) for label, probs in grouped_probs.items()}
    stds = {label: np.std(probs) for label, probs in grouped_probs.items()}

    # Select samples whose predicted prob for their label exceeds threshold
    selection = [
        i for i in range(len(X))
        if e[i][y[i]] >= (means[y[i]] + K * stds[y[i]])
    ]

    return np.array(selection, dtype=np.int32)


def update_us(
    u: npt.NDArray[np.float32],
    e: npt.NDArray[np.float32],
    p: npt.NDArray[np.float32],
    L: float,
) -> npt.NDArray[np.float32]:
    """
    Updates the logits (pre-sigmoid confidence values) based on the difference
    between ensemble predictions and current soft labels.

    Args:
        u: Current logits (pre-sigmoid) of shape (n_samples, n_classes).
        e: Predicted class probabilities from ensemble of shape (n_samples, n_classes).
        p: Current soft label probabilities of shape (n_samples, n_classes).
        L: Learning rate for the update step.

    Returns:
        Updated logits of shape (n_samples, n_classes).
    """

    return u + L * (e - p)


def update_probabilities(u: npt.NDArray[np.float32], B: float) -> npt.NDArray[np.float32]:
    """
    Updates label probability values using a sigmoid transformation.

    Args:
        u: A (n_samples, n_classes) matrix of confidence logits.
        B: A positive scaling factor to control the sharpness of the sigmoid.

    Returns:
        A (n_samples, n_classes) matrix of normalized class probabilities.
    """

    # Apply sigmoid elementwise with broadcasting
    p = (1 + np.tanh(u / B)) / 2

    # Normalize rows to sum to 1, with numerical stability
    row_sums = np.sum(p, axis=1, keepdims=True) + 1e-8  # avoid division by zero
    p_normalized = p / row_sums

    return p_normalized.astype(np.float32)

def update_labels(p: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Updates hard labels based on the current soft label probabilities.

    Args:
        p: A (n_samples, n_classes) matrix of class probabilities.

    Returns:
        A (n_samples,) array of hard labels, where each label is the index of the
        class with the highest probability.
    """

    return np.argmax(p, axis=1)


def sigmoid(x, b):
    """
    Sigmoid function with a scaling parameter `b` to control steepness.

    Args:
        x: Input value (float).
        b: Scaling factor (positive float). Smaller values make the transition sharper.

    Returns:
        A value in the range (0, 1) representing the sigmoid output.
    """

    return (1 + math.tanh(x / b)) / 2


def inv_sigmoid(x, b):
    """
    Inverse of the scaled sigmoid function.

    Args:
        x: A float value in (0, 1) representing a probability.
        b: Scaling factor used in the original sigmoid.

    Returns:
        The corresponding pre-sigmoid (logit-like) value.

    Notes:
        - Clamps `x` to avoid domain errors in atanh (which is only defined on (-1, 1)).
        - Assumes the original sigmoid was computed as: (1 + tanh(u / b)) / 2.
    """

    x = max(0.0001, min(0.9999, x))  # Clamp x to avoid out-of-domain errors
    return math.atanh(2 * x - 1) * b