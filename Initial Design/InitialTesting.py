import numpy as np
from InitialAdjustedRandomForest import train, hard_train
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from scipy.stats import sem
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Callable, Tuple, Sequence,Optional


def plot_with_error_band(
    x: Sequence[float],
    y_mean: Sequence[float],
    y_se: Sequence[float],
    label: str,
    color: str,
    linestyle: str = "-",
    alpha: float = 0.2
) -> None:
    """
    Plots a line with a shaded error band using mean and standard error.

    Args:
        x: X-axis values.
        y_mean: Mean values to plot (e.g., accuracy across trials).
        y_se: Standard error values for shaded region.
        label: Label to show in legend.
        color: Line and fill color.
        linestyle: Line style (e.g., '-', '--').
        alpha: Transparency of the shaded band.
    """
    plt.plot(x, y_mean, label=label, color=color, linestyle=linestyle, linewidth=2)

    # Fill region between mean ± standard error
    plt.fill_between(
        x,
        np.array(y_mean) - np.array(y_se),
        np.array(y_mean) + np.array(y_se),
        alpha=alpha,
        color=color
    )

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

def run_relabelling_experiment(
    dataset,
    forest_class: type[ClassifierMixin],
    noise_func: Callable[[np.ndarray, float], np.ndarray],
    n_estimators: int = 10,
    trials: int = 35,
    resolution: int = 20,
    test_size: float = 0.25,
    methods: Tuple[str, ...] = ("standard", "control"),
    relabelling: bool = True,
    initial_certainty: float = 0.95,
    K: float = 0.5,
    L: float = 0.01,
    bootstrap: bool = True,
    relabel_bunch: int = 1,
    seed: int = 42,
    hard: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs relabelling experiments over multiple trials and noise levels.

    Args:
        dataset: Dataset with `.data` and `.target`.
        forest_class: A classifier class (e.g., RandomForestClassifier).
        noise_func: Function that applies label noise.
        n_estimators: Number of estimators for the forest.
        trials: Number of independent runs per noise level.
        resolution: Number of noise steps between 0 and 1.
        test_size: Proportion of data used for testing.
        methods: Tuple of methods to compare ("standard", "control", etc.).
        relabelling: Whether to enable relabelling logic.
        initial_certainty: Soft-label initial confidence.
        K: Confidence threshold for sample filtering.
        L: Learning rate for updating logits.
        bootstrap: Whether to use bootstrapping in training.
        relabel_bunch: How many estimators per relabelling iteration.
        seed: Random seed for reproducibility.
        hard: If True, uses hard_train() instead of soft relabelling.

    Returns:
        accuracies_all: Accuracy results for each method (n_methods × resolution × trials).
        relabelling_f1_all: F1 scores for relabelling correctness.
        relabelling_acc_all: Accuracy of relabelling compared to ground truth.
        noise_levels: Array of tested noise levels.
    """
    n_methods = len(methods)
    accuracies_all = np.zeros((n_methods, resolution, trials))
    relabelling_f1_all = np.zeros((2, resolution, trials))
    relabelling_acc_all = np.zeros((2, resolution, trials))
    noise_levels = np.linspace(0, 1, resolution)

    for i, noise_rate in enumerate(noise_levels):
        for trial in range(trials):
            # Split the dataset
            X_train, X_test, y_train, y_test = train_test_split(
                dataset.data, dataset.target,
                test_size=test_size,
                random_state=seed + trial
            )

            # Apply noise to training labels
            y_mislabelled = noise_func(np.copy(y_train), noise_rate, seed=seed + trial)

            # Method 0: Adjusted Random Forest with relabelling
            if "standard" in methods:
                if not hard:
                    rf, corrected_y, _ = train(
                        forest_class, X_train, y_mislabelled, np.unique(dataset.target),
                        n_estimators=n_estimators,
                        relabelling=relabelling,
                        initial_certainty=initial_certainty,
                        K=K, L=L,
                        bootstrap=bootstrap,
                        relabel_bunch=relabel_bunch
                    )
                else:
                    rf, corrected_y = hard_train(
                        forest_class, X_train, y_mislabelled, np.unique(dataset.target),
                        n_estimators=n_estimators,
                        relabelling=relabelling,
                        initial_certainty=initial_certainty,
                        K=K, L=L,
                        bootstrap=bootstrap
                    )

                # Evaluate relabelling performance
                y_pred = rf.predict(X_test)
                relabelling_f1_all[0, i, trial], relabelling_acc_all[0, i, trial] = relabelling_f1(
                    y_train, y_mislabelled, corrected_y
                )
                accuracies_all[0, i, trial] = metrics.accuracy_score(y_test, y_pred)

            # Method 1: Standard classifier with no relabelling (control)
            if "control" in methods:
                rf = forest_class(n_estimators=n_estimators, criterion="entropy")
                rf.fit(X_train, y_mislabelled)
                y_pred = rf.predict(X_test)
                accuracies_all[1, i, trial] = metrics.accuracy_score(y_test, y_pred)

    return accuracies_all, relabelling_f1_all, relabelling_acc_all, noise_levels

from typing import Tuple
from sklearn import metrics
import numpy as np

def relabelling_f1(
    y_true: np.ndarray,
    y_mislabelled: np.ndarray,
    y_corrected: np.ndarray
) -> Tuple[float, float]:
    """
    Computes F1 score and relabelling accuracy based on how well noisy labels were corrected.

    Args:
        y_true: Ground-truth labels.
        y_mislabelled: Noisy labels (input to model).
        y_corrected: Labels after relabelling.

    Returns:
        F1 score comparing noise vs change masks,
        Accuracy of corrected noisy samples.
    """
    noise_mask = y_mislabelled != y_true          # Identify noisy samples
    correct_mask = y_true == y_corrected          # Samples correctly relabelled
    change_mask = y_mislabelled != y_corrected    # Samples that were relabelled
    relabel_correct = noise_mask & correct_mask   # Noisy samples correctly relabelled

    if noise_mask.sum() == 0 and change_mask.sum() == 0:
        return 1.0, 1.0  # Perfect case, nothing needed, nothing changed
    elif noise_mask.sum() == 0:
        return 0.0, 0.0  # No noise but changes occurred — undesirable
    else:
        f1 = metrics.f1_score(noise_mask, change_mask)
        acc = sum(relabel_correct) / (sum(noise_mask) + 1e-8)  # Accuracy over noisy
        return f1, acc


def process_experiment_results(
    accuracies_all: np.ndarray,
    relabelling_f1_all: Optional[np.ndarray] = None,
    relabelling_acc_all: Optional[np.ndarray] = None,
    resolution: int = 20,
    test_size: float = 0.25,
    dataset_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Processes raw experimental outputs to compute mean and standard error.

    Args:
        accuracies_all: Array of shape (n_methods, resolution, trials).
        relabelling_f1_all: Optional array for relabelling F1 results.
        relabelling_acc_all: Optional array for relabelling accuracy.
        resolution: Number of tested noise levels.
        test_size: Proportion of test set (used for context only).
        dataset_size: Total sample size before split (unused here, placeholder).

    Returns:
        Tuple containing:
            - Mean and SE for accuracy
            - Mean and SE for relabelling F1 (or None)
            - Mean and SE for relabelling accuracy (or None)
            - Noise level grid (np.ndarray)
    """
    # Compute mean and SE for accuracy
    accuracies_all = np.array(accuracies_all)
    accuracies_mean = accuracies_all.mean(axis=2)
    accuracies_se = sem(accuracies_all, axis=2)

    # Initialize relabelling results
    relabelling_f1_success = relabelling_f1_se = None
    relabelling_acc_success = relabelling_acc_se = None

    if relabelling_f1_all is not None:
        relabelling_f1_all = np.array(relabelling_f1_all)
        relabelling_f1_success = relabelling_f1_all.mean(axis=2)
        relabelling_f1_se = sem(relabelling_f1_all, axis=2)

    if relabelling_acc_all is not None:
        relabelling_acc_all = np.array(relabelling_acc_all)
        relabelling_acc_success = relabelling_acc_all.mean(axis=2)
        relabelling_acc_se = sem(relabelling_acc_all, axis=2)

    # Generate the x-axis of noise levels
    noise_levels = np.linspace(0, 1, resolution)

    return (
        accuracies_mean, accuracies_se,
        relabelling_f1_success, relabelling_f1_se,
        relabelling_acc_success, relabelling_acc_se,
        noise_levels
    )


def plot_rf_decision_boundary(
    rf: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    resolution: int = 100,
    class_index: int = 1,
    cmap: str = 'bwr'
) -> None:
    """
    Visualizes a RandomForest decision boundary on 2D data.

    Args:
        rf: Trained RandomForestClassifier.
        X: Feature matrix (2D only), shape (n_samples, 2).
        y: Labels corresponding to X.
        resolution: Grid resolution for heatmap.
        class_index: Which class probability to visualize.
        cmap: Colormap for contour heatmap.
    """
    if X.shape[1] != 2:
        raise ValueError("This function only supports 2D input features.")

    # Compute bounds of the plot
    x_min, x_max = X[:, 0].min() * 0.9, X[:, 0].max() * 1.1
    y_min, y_max = X[:, 1].min() * 0.9, X[:, 1].max() * 1.1

    # Create a mesh grid
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict class probabilities for each point in the grid
    probs = rf.predict_proba(grid)[:, class_index]
    probs = probs.reshape(xx.shape)

    # Create contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xx, yy, probs, levels=100, cmap=cmap, alpha=0.7)
    plt.colorbar(contour, label=f'P(class={class_index})')

    # Overlay original points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k',
                cmap=ListedColormap(['#1f77b4', '#ff7f0e']))
    plt.title("Random Forest Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()