'''
Additional implementations of the adjusted random forest for outputting particular 
data from training which would not otherwise be provided
'''

import math
import copy
import numpy as np
import numpy.typing as npt
from typing import Union
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.utils import shuffle
from AdjustedRandomForest import slice_prediction, update_labels, update_probabilities, update_us, inv_sigmoid, select_training_data


def train_output_all_iterations(
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
) -> tuple[RandomForestClassifier | ExtraTreesClassifier, npt.NDArray[np.int32]]:

    forests = []
    ys_ordered = []

    labels = labels = sorted(np.unique(y))
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
    forest = ensemble(criterion="entropy", bootstrap=False, warm_start=True)
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

        forests.append(copy.deepcopy(forest))
        ys_ordered.append(np.copy(y_ordered))  # Use np.copy to avoid later mutation

    return forests, ys_ordered

def train_log_no_threshold_samples(
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
    no_threshold_samples = 0
    labels = labels = sorted(np.unique(y))
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
    forest = ensemble(criterion="entropy", bootstrap=False, warm_start=True)
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
            no_threshold_samples += len(set(selection) & set(indices))

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

    return forest, y_ordered, no_threshold_samples/(n_estimators*iterations)