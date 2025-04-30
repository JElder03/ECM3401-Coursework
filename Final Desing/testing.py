import numpy as np
from sklearn.utils import Bunch
from typing import Callable, Union, Tuple
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from AdjustedRandomForest import train
import Testing_ARFs 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats import sem
import matplotlib.pyplot as plt


def run_noise_level_experiment(
    dataset,
    forest_class: Union[type[RandomForestClassifier], type[ExtraTreesClassifier]],
    noise_func: Callable[[np.ndarray, float], np.ndarray],
    n_estimators: int = 10,
    trials: int = 35,
    resolution: int = 20,
    test_size: float = 0.25,
    iterations: int = 10,
    control: bool = True,
    relabelling: bool = True,
    bootstrapping: bool = False,
    K: float = 0.5,
    L: float = 0.01,
    initial_certainty: float = 0.95,
    max_features='sqrt',
    noise_ratio=None,
    clf=None,
    unique_pairs=None,
    seed=42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs relabelling experiments over multiple trials and noise levels.

    Args:
        dataset: Object with `.data` and `.target` (e.g., sklearn.datasets).
        forest_class: The ensemble classifier type (RandomForest or ExtraTrees).
        noise_func: Function to apply label noise.
        n_estimators: Number of estimators per iteration (used by relabelling method).
        trials: Number of independent trials per noise level.
        resolution: Number of evenly spaced noise levels from 0 to 1.
        test_size: Fraction of data reserved for testing.
        iterations: Number of iterative training steps for the relabelling method.
        control: If True, runs a baseline classifier without relabelling.
        relabelling: Enables relabelling in the main method if True.
        bootstrapping: Whether to use bootstrapping in relabelling.
        K: Confidence threshold for sample selection.
        L: Learning rate for logit updates.
        initial_certainty: Initial certainty used in soft label creation.
        max_features: Feature subset strategy (e.g., 'sqrt').
        noise_ratio: Optionally scale noise per class or group.
        clf: Optional auxiliary model (e.g., NNAR) for label noise injection.
        unique_pairs: Optional label groupings for structured noise.
        seed: Random seed for reproducibility.

    Returns:
        accuracies_all: Accuracy array (2, resolution, trials) for method and control.
        auc_all: AUC array (2, resolution, trials) for method and control.
        relabelling_f1_all: F1 scores between true and relabelled training labels.
        relabelling_acc_all: Accuracy scores for relabelling.
        noise_levels: Array of evaluated noise levels.
    """

    # Pre-allocate results
    accuracies_all = np.zeros((2, resolution, trials))
    auc_all = np.zeros((2, resolution, trials))
    relabelling_f1_all = np.zeros((1, resolution, trials))
    relabelling_acc_all = np.zeros((1, resolution, trials))
    noise_levels = np.linspace(0, 1, resolution)
    noise_ratio = 1 if noise_ratio is None else np.array(noise_ratio)

    for i, noise_rate in enumerate(noise_levels):
        for trial in range(trials):
            # Split dataset and apply label noise
            X_train, X_test, y_train, y_test = train_test_split(
                dataset.data, dataset.target, test_size=test_size,
                stratify=dataset.target, random_state=seed + trial
            )

            # Apply noise depending on experiment type
            if clf:
                y_mislabelled = noise_func(X_train, np.copy(y_train), clf, noise_rate, seed=seed + trial)
            elif unique_pairs:
                y_mislabelled = noise_func(np.copy(y_train), noise_rate * noise_ratio, seed=seed + trial, unique_pairs=unique_pairs)
            else:
                y_mislabelled = noise_func(np.copy(y_train), noise_rate * noise_ratio, seed=seed + trial)

            # === Method 0: Reweighting + Relabelling Ensemble ===
            rf, y_corrected = train(
                forest_class, X_train, y_mislabelled,
                n_estimators=n_estimators,
                iterations=iterations,
                bootstrapping=bootstrapping,
                relabelling=relabelling,
                K=K, L=L,
                initial_certainty=initial_certainty,
                labels=np.unique(dataset.target),
                max_features=max_features
            )

            y_pred = rf.predict(X_test)
            y_score = rf.predict_proba(X_test)

            # Evaluate method
            accuracies_all[0, i, trial] = metrics.accuracy_score(y_test, y_pred)
            if len(np.unique(y_test)) <= 2:
                auc_all[0, i, trial] = metrics.roc_auc_score(y_test, y_score[:, 1], labels=np.unique(dataset.target))
            else:
                auc_all[0, i, trial] = metrics.roc_auc_score(y_test, y_score, multi_class='ovr', labels=np.unique(dataset.target))

            # Relabelling quality
            relabelling_f1_all[0, i, trial], relabelling_acc_all[0, i, trial] = relabelling_f1(y_train, y_mislabelled, y_corrected)

            # === Method 1: Control - Standard Ensemble Without Relabelling ===
            if control:
                # Ensure all classes present (for AUC stability)
                missing = set(dataset.target) - set(y_mislabelled)
                if missing:
                    dummy_X = np.zeros((len(missing), X_train.shape[1]))
                    dummy_y = np.array(list(missing))
                    X_train = np.vstack([X_train, dummy_X])
                    y_mislabelled = np.concatenate([y_mislabelled, dummy_y])
                    sample_weight = np.ones(len(y_mislabelled))
                    sample_weight[-len(missing):] = 0  # Dummy samples don’t affect training
                else:
                    sample_weight = np.ones(len(y_mislabelled))

                # Train a standard ensemble (no iterative relabelling)
                rf = forest_class(
                    n_estimators=n_estimators * iterations,
                    random_state=seed + trial
                )
                rf.fit(X_train, y_mislabelled, sample_weight=sample_weight)

                y_pred = rf.predict(X_test)
                y_score = rf.predict_proba(X_test)

                # Evaluate control method
                accuracies_all[1, i, trial] = metrics.accuracy_score(y_test, y_pred)
                if len(np.unique(y_test)) <= 2:
                    auc_all[1, i, trial] = metrics.roc_auc_score(y_test, y_score[:, 1], labels=np.unique(dataset.target))
                else:
                    auc_all[1, i, trial] = metrics.roc_auc_score(y_test, y_score, multi_class='ovr', labels=np.unique(dataset.target))

    return accuracies_all, auc_all, relabelling_f1_all, relabelling_acc_all, noise_levels


def run_parameter_sweep_experiment(
    dataset,
    forest_class: Union[type[RandomForestClassifier], type[ExtraTreesClassifier]],
    noise_func: Callable[[np.ndarray, float], np.ndarray],
    param_name: str,
    param_values: list,
    noise_rate: float = 0.4,
    n_estimators: int = 10,
    trials: int = 20,
    test_size: float = 0.25,
    iterations: int = 10,
    control: bool = True,
    relabelling: bool = True,
    bootstrapping: bool = False,
    K: float = 0.5,
    L: float = 0.01,
    B: float = 0.02,
    initial_certainty: float = 0.95,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates how changing a specific parameter affects relabelling performance at a fixed noise level.

    Args:
        dataset: Dataset with `.data` and `.target`.
        forest_class: Ensemble classifier type (e.g. RandomForestClassifier).
        noise_func: Function that applies label noise to the training set.
        param_name: Name of the training parameter to vary (e.g. 'K', 'L', etc.).
        param_values: List of values to try for the selected parameter.
        noise_rate: Fixed noise level to test across all parameter values.
        n_estimators: Number of estimators per iteration.
        trials: Number of random trials to average results over.
        test_size: Proportion of the dataset used as a test split.
        iterations: Number of relabelling iterations.
        control: Whether to run a baseline control model without relabelling.
        relabelling: Whether to enable relabelling in the main method.
        bootstrapping: Whether to use bootstrapping in relabelling.
        K, L, B, initial_certainty: Relabelling algorithm hyperparameters.
        seed: Random seed for reproducibility.

    Returns:
        accuracies_all: Shape (2, len(param_values), trials), accuracy for relabelling and control.
        auc_all: Shape (2, len(param_values), trials), AUC for relabelling and control.
        relabelling_f1_all: Shape (1, len(param_values), trials), F1 between true and relabelled labels.
        relabelling_acc_all: Shape (1, len(param_values), trials), relabelling accuracy.
        param_values: Array of values tested for the selected parameter.
    """

    n_params = len(param_values)

    # Initialize result containers
    accuracies_all = np.zeros((2, n_params, trials))
    auc_all = np.zeros((2, n_params, trials))
    relabelling_f1_all = np.zeros((1, n_params, trials))
    relabelling_acc_all = np.zeros((1, n_params, trials))

    for i, param_value in enumerate(param_values):
        for trial in range(trials):
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                dataset.data, dataset.target, test_size=test_size,
                stratify=dataset.target, random_state=seed + trial
            )

            # Apply noise to labels
            y_mislabelled = noise_func(np.copy(y_train), noise_rate, seed=seed + trial)

            # Prepare training parameters, injecting the current sweep value
            train_kwargs = dict(
                n_estimators=n_estimators,
                iterations=iterations,
                bootstrapping=bootstrapping,
                relabelling=relabelling,
                K=K,
                L=L,
                B=B,
                initial_certainty=initial_certainty,
            )
            train_kwargs[param_name] = param_value

            # Method 0: Relabelling model
            rf, y_corrected = train(
                forest_class, X_train, y_mislabelled,
                **train_kwargs
            )

            # Evaluate
            y_pred = rf.predict(X_test)
            y_score = rf.predict_proba(X_test)

            accuracies_all[0, i, trial] = metrics.accuracy_score(y_test, y_pred)
            if len(np.unique(y_test)) <= 2:
                auc_all[0, i, trial] = metrics.roc_auc_score(y_test, y_score[:, 1], labels=np.unique(dataset.target))
            else:
                auc_all[0, i, trial] = metrics.roc_auc_score(y_test, y_score, multi_class='ovr', labels=np.unique(dataset.target))

            # Evaluate relabelling quality
            relabelling_f1_all[0, i, trial], relabelling_acc_all[0, i, trial] = relabelling_f1(
                y_train, y_mislabelled, y_corrected
            )

    # Method 1: Control model (no relabelling, single model reused across param_values)
    if control:
        for trial in range(trials):
            rf = forest_class(n_estimators=n_estimators * iterations, random_state = seed + trial)
            rf.fit(X_train, y_mislabelled)

            y_pred = rf.predict(X_test)
            y_score = rf.predict_proba(X_test)

            accuracies_all[1, :, trial] = metrics.accuracy_score(y_test, y_pred)

            if len(np.unique(y_test)) <= 2:
                auc_all[1, :, trial] = metrics.roc_auc_score(y_test, y_score[:, 1], labels=np.unique(dataset.target))
            else:
                auc_all[1, :, trial] = metrics.roc_auc_score(y_test, y_score, multi_class='ovr', labels=np.unique(dataset.target))

    return accuracies_all, auc_all, relabelling_f1_all, relabelling_acc_all, np.array(param_values)


def run_no_threshold_samples_experiment(
    dataset,
    forest_class: Union[type[RandomForestClassifier], type[ExtraTreesClassifier]],
    noise_func: Callable[[np.ndarray, float], np.ndarray],
    param_name: str,
    param_values: list,
    noise_rate: float = 0.4,
    n_estimators: int = 10,
    trials: int = 20,
    test_size: float = 0.25,
    iterations: int = 10,
    control: bool = True,
    relabelling: bool = True,
    bootstrapping: bool = False,
    K: float = 0.5,
    L: float = 0.01,
    B: float = 0.02,
    initial_certainty: float = 0.95,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates how often no samples are selected during training due to thresholding,
    across different values of a specified parameter, under fixed label noise.

    Args:
        dataset: Dataset object with `.data` and `.target`.
        forest_class: Classifier class (e.g. RandomForestClassifier).
        noise_func: Callable that applies noise to labels.
        param_name: Name of the parameter to vary (e.g. 'K').
        param_values: List of values for the parameter being swept.
        noise_rate: Fixed noise level to test with.
        n_estimators: Number of estimators per iteration in the relabelling method.
        trials: Number of independent trials per parameter value.
        test_size: Proportion of data reserved for testing.
        iterations: Number of training iterations.
        control: Placeholder flag (not used in this function).
        relabelling: Whether to apply relabelling in the training process.
        bootstrapping: Whether bootstrapping is used in the final iteration.
        K, L, B, initial_certainty: Hyperparameters for the relabelling algorithm.
        seed: Random seed for reproducibility.

    Returns:
        accuracies_all: Shape (1, len(param_values), trials), classification accuracy.
        relabelling_f1_all: Shape (1, len(param_values), trials), F1 scores for relabelling.
        relabelling_acc_all: Shape (1, len(param_values), trials), relabelling accuracy.
        param_values: Numpy array of the parameter values tested.
        no_samples_count_all: Shape (1, len(param_values), trials), number of iterations
                              where no samples passed the confidence threshold.
    """

    n_params = len(param_values)

    # Initialize result containers
    no_samples_count_all = np.zeros((1, n_params, trials))
    accuracies_all = np.zeros((1, n_params, trials))
    relabelling_f1_all = np.zeros((1, n_params, trials))
    relabelling_acc_all = np.zeros((1, n_params, trials))

    for i, param_value in enumerate(param_values):
        for trial in range(trials):
            # Split and apply label noise
            X_train, X_test, y_train, y_test = train_test_split(
                dataset.data, dataset.target,
                test_size=test_size,
                stratify=dataset.target,
                random_state=seed + trial
            )
            y_mislabelled = noise_func(np.copy(y_train), noise_rate, seed=seed + trial)

            # Inject the sweeping parameter value
            train_kwargs = dict(
                n_estimators=n_estimators,
                iterations=iterations,
                bootstrapping=bootstrapping,
                relabelling=relabelling,
                K=K,
                L=L,
                B=B,
                initial_certainty=initial_certainty,
            )
            train_kwargs[param_name] = param_value

            # Call custom training function that also logs no-sample counts
            rf, y_corrected, no_samples_count = Testing_ARFs.train_log_no_threshold_samples(
                forest_class, X_train, y_mislabelled, **train_kwargs
            )

            # Evaluate predictions
            y_pred = rf.predict(X_test)

            # Record results
            no_samples_count_all[0, i, trial] = no_samples_count
            accuracies_all[0, i, trial] = metrics.accuracy_score(y_test, y_pred)
            relabelling_f1_all[0, i, trial], relabelling_acc_all[0, i, trial] = relabelling_f1(
                y_train, y_mislabelled, y_corrected
            )

    return accuracies_all, relabelling_f1_all, relabelling_acc_all, np.array(param_values), no_samples_count_all


def run_iteration_sweep_experiment(
    dataset,
    forest_class: Union[type[RandomForestClassifier], type[ExtraTreesClassifier]],
    noise_func: Callable[[np.ndarray, float], np.ndarray],
    noise_rate: float = 0.4,
    n_estimators: int = 10,
    trials: int = 20,
    test_size: float = 0.25,
    iterations: int = 10,
    relabelling: bool = True,
    bootstrapping: bool = False,
    K: float = 0.5,
    L: float = 0.01,
    initial_certainty: float = 0.95,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Evaluates model accuracy and relabelling quality across training iterations.

    This function runs a relabelling ensemble training algorithm over a fixed
    number of iterations, returning metrics at each step.

    Args:
        dataset: Dataset with `.data` and `.target`.
        forest_class: Classifier class (e.g., RandomForestClassifier).
        noise_func: Function that introduces label noise.
        noise_rate: Fraction of training labels to corrupt.
        n_estimators: Number of estimators per iteration.
        trials: Number of independent trials.
        test_size: Proportion of data used for testing.
        iterations: Number of relabelling/reweighting iterations.
        relabelling: Whether to apply relabelling logic.
        bootstrapping: Whether bootstrapping is used during training.
        K, L, initial_certainty: Algorithm hyperparameters.
        seed: Random seed for reproducibility.

    Returns:
        accuracies_all:      (1, iterations, trials) array of classification accuracies.
        relabelling_f1_all:  (1, iterations, trials) F1 scores of relabelling.
        relabelling_acc_all: (1, iterations, trials) Accuracy of relabelling.
        iteration_range:     List of iteration indices (1-based).
    """

    # Preallocate result arrays
    accuracies_all = np.zeros((1, iterations, trials))
    relabelling_f1_all = np.zeros((1, iterations, trials))
    relabelling_acc_all = np.zeros((1, iterations, trials))

    for trial in range(trials):
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target,
            test_size=test_size,
            stratify=dataset.target,
            random_state=seed + trial
        )

        # Apply label noise
        y_mislabelled = noise_func(np.copy(y_train), noise_rate, seed=seed + trial)

        # Run training and collect models + relabelled labels per iteration
        rfs, ys_corrected = Testing_ARFs.train_output_all_iterations(
            forest_class, X_train, y_mislabelled,
            n_estimators=n_estimators,
            iterations=iterations,
            bootstrapping=bootstrapping,
            relabelling=relabelling,
            K=K,
            L=L,
            initial_certainty=initial_certainty
        )

        # Evaluate each iteration
        for i, rf in enumerate(rfs):
            y_pred = rf.predict(X_test)
            accuracies_all[0, i, trial] = metrics.accuracy_score(y_test, y_pred)
            relabelling_f1_all[0, i, trial], relabelling_acc_all[0, i, trial] = relabelling_f1(
                y_train, y_mislabelled, ys_corrected[i]
            )

    # Return performance traces over iterations
    return accuracies_all, relabelling_f1_all, relabelling_acc_all, list(range(1, iterations + 1))


def relabelling_f1(y_true, y_mislabelled, y_corrected):
    # Identify noisy labels: where the original label doesn't match the noisy one
    noise_mask = y_mislabelled != y_true

    # Identify correct relabels: where corrected label matches the original
    correct_mask = y_true == y_corrected

    # Identify changes made: where corrected label differs from the noisy one
    change_mask = y_mislabelled != y_corrected

    # Identify successful relabels: labels that were noisy and were fixed
    relabel_correct = noise_mask & correct_mask

    # Handle edge cases
    if noise_mask.sum() == 0 and change_mask.sum() == 0:
        return 1.0, 1.0  # No noise and no relabelling = perfect
    elif noise_mask.sum() == 0:
        return 0.0, 0.0  # No noise but changes occurred = undesired
    else:
        # Return relabelling F1 and correction accuracy over noisy samples
        return metrics.f1_score(noise_mask, change_mask), sum(relabel_correct) / (sum(noise_mask) + 1e-8)


def process_experiment_result(results: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes mean and standard error across trials for experimental results.

    Args:
        results: Array of shape (n_methods, resolution, trials) or similar.

    Returns:
        mean: Array of mean values across trials.
        se:   Array of standard errors across trials.
    """
    results = np.array(results)
    return results.mean(axis=2), sem(results, axis=2)

    
def plot_with_error_band(x, y_mean, y_se, label, color, linestyle="-", alpha=0.2):
    """
    Plots a line graph with shaded standard error bands.

    Args:
        x:         X-axis values.
        y_mean:    Mean values to plot.
        y_se:      Standard error values for the shaded region.
        label:     Label for legend.
        color:     Line and fill color.
        linestyle: Line style (e.g., '-', '--').
        alpha:     Opacity of shaded area.
    """
    import matplotlib.pyplot as plt

    # Line for mean
    plt.plot(x, y_mean, label=label, color=color, linestyle=linestyle, linewidth=2)

    # Shaded error band
    plt.fill_between(x, y_mean - y_se, y_mean + y_se, alpha=alpha, color=color)

    # Optional aesthetics
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()


def plot_heatmap(matrix, x_axis, y_axis, label, x_title, y_title):
    """
    Plots a heatmap of a performance matrix over two parameter axes.

    Args:
        matrix: 2D array with shape (len(y_axis), len(x_axis)).
        x_axis: List of values for the x-axis (e.g., n_estimators).
        y_axis: List of values for the y-axis (e.g., iterations).
        label:  Colorbar label.
        x_title: X-axis title.
        y_title: Y-axis title.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()

    # Display matrix as an image
    cax = ax.imshow(matrix, interpolation='nearest', cmap='viridis', origin='lower')

    # Label axes with parameter values
    ax.set_xticks(np.arange(len(x_axis)))
    ax.set_xticklabels(x_axis)

    ax.set_yticks(np.arange(len(y_axis)))
    ax.set_yticklabels(y_axis)

    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)

    # Add colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label(label)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def load_gmm5():
    """
    Loads the GMM5 synthetic dataset from local text files.

    Returns:
        A sklearn-style Bunch with:
            - data:     Feature matrix (n_samples, 2)
            - target:   Class labels (0 or 1)
            - feature_names, target_names, DESCR
    """
    import numpy as np
    from sklearn.utils import Bunch

    # Load test and training data
    data1 = np.loadtxt("../Data/gmm5test.txt")
    data2 = np.loadtxt("../Data/gmm5train.txt")

    # Combine and shuffle
    dataset = np.vstack((data1, data2))
    np.random.seed(42)
    np.random.shuffle(dataset)

    # Extract features and labels
    X = dataset[:, :2]
    y = dataset[:, 2].astype(int)

    # Wrap in sklearn-compatible format
    sklearn_like = {
        'data': X,
        'target': y,
        'feature_names': ['feature_1', 'feature_2'],
        'target_names': [0, 1],
        'DESCR': 'GMM5',
    }

    return Bunch(**sklearn_like)