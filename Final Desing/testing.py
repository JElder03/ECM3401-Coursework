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
    max_features = 'sqrt',
    noise_ratio = None,
    clf = None,
    seed = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs relabelling experiments over multiple trials and varying noise levels.

    Args:
        dataset: Object with `.data` and `.target` (e.g. from sklearn.datasets.load_*).
        forest_class: The classifier type (e.g., RandomForestClassifier).
        noise_func: Function to apply label noise. Takes (labels, noise_rate).
        n_estimators: Number of estimators per iteration (for relabelling methods).
        trials: Number of independent trials per noise level.
        resolution: Number of noise steps between 0 and 1.
        test_size: Proportion of data to reserve for testing.
        iterations: Number of iterations in the reweighting/relabeling scheme.
        methods: Tuple of method names to evaluate.
        relabelling: If True, enables relabelling in the reweighting algorithm.
        K: Confidence threshold for sample selection.
        L: Learning rate for updating logits.
        initial_certainty: Initial label confidence for one-hot smoothing.

    Returns:
        accuracies_all: Array of shape (n_methods, resolution, trials).
        relabelling_all: Array of relabelling counts for first two methods.
        noise_levels: Array of noise rates.
    """

    accuracies_all = np.zeros((2, resolution, trials))
    auc_all = np.zeros((2, resolution, trials))

    relabelling_f1_all = np.zeros((1, resolution, trials))
    relabelling_acc_all = np.zeros((1, resolution, trials))

    noise_levels = np.linspace(0, 1, resolution)
    noise_ratio = 1 if noise_ratio is None else np.array(noise_ratio)

    for i, noise_rate in enumerate(noise_levels):
        for trial in range(trials):
            # Split and corrupt data
            X_train, X_test, y_train, y_test = train_test_split(
                dataset.data, dataset.target, test_size=test_size, stratify=dataset.target, random_state = seed + trial
            )

            if clf:
                #NNAR
                y_mislabelled = noise_func(X_train, np.copy(y_train), clf, noise_rate, seed = seed + trial)
            else:
                y_mislabelled = noise_func(np.copy(y_train), noise_rate * noise_ratio, seed = seed + trial)

            # Method 0: standard relabelling
            rf, y_corrected = train(
                forest_class, X_train, y_mislabelled,
                n_estimators=n_estimators,
                iterations=iterations,
                bootstrapping = bootstrapping,
                relabelling=relabelling,
                K=K, L=L,
                initial_certainty=initial_certainty,
                labels = np.unique(dataset.target),
                max_features= max_features
            )

            y_pred = rf.predict(X_test)
            y_score = rf.predict_proba(X_test)

            accuracies_all[0, i, trial] = metrics.accuracy_score(y_test, y_pred)
            if len(np.unique(dataset.target)) == 2:
                auc_all[0, i, trial] = metrics.roc_auc_score(y_test, y_score[:, 1], labels=np.unique(dataset.target))
            else:
                auc_all[0, i, trial] = metrics.roc_auc_score(y_test, y_score, multi_class='ovr', labels=np.unique(dataset.target))
            relabelling_f1_all[0, i, trial], relabelling_acc_all[0, i, trial] = relabelling_f1(y_train, y_mislabelled, y_corrected)

            # Method 1: control (no relabelling, full estimators)
            if control:
                missing = set(dataset.target) - set(y_mislabelled)
                if missing:
                    dummy_X = np.zeros((len(missing), X_train.shape[1]))
                    dummy_y = np.array(list(missing))
                    # Append dummy samples
                    X_train = np.vstack([X_train, dummy_X])
                    y_mislabelled = np.concatenate([y_mislabelled, dummy_y])
                    sample_weight = np.ones(len(y_mislabelled))
                    sample_weight[-len(missing):] = 0
                else:
                    sample_weight = np.ones(len(y_mislabelled))


                rf = forest_class(
                    n_estimators=n_estimators * iterations,
                    random_state = seed + trial
                )
                rf.fit(X_train, y_mislabelled, sample_weight=sample_weight)

                y_pred = rf.predict(X_test)
                y_score = rf.predict_proba(X_test)

                accuracies_all[1, i, trial] = metrics.accuracy_score(y_test, y_pred)
                if len(np.unique(dataset.target)) == 2:
                    auc_all[0, i, trial] = metrics.roc_auc_score(y_test, y_score[:, 1], labels=np.unique(dataset.target))
                else:
                    auc_all[0, i, trial] = metrics.roc_auc_score(y_test, y_score, multi_class='ovr', labels=np.unique(dataset.target))

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates the effect of changing a specific training parameter at a fixed noise level.

    Args:
        param_name: Name of the parameter to vary ("K", "L", "initial_certainty", etc.)
        param_values: List of values to try for that parameter
        noise_rate: The fixed noise rate to test against

    Returns:
        accuracies_all: shape (2, len(param_values), trials)
        relabelling_f1_all: shape (1, len(param_values), trials)
        relabelling_acc_all: shape (1, len(param_values), trials)
        param_values: values that were tested
    """

    n_params = len(param_values)
    accuracies_all = np.zeros((2, n_params, trials))
    relabelling_f1_all = np.zeros((1, n_params, trials))
    relabelling_acc_all = np.zeros((1, n_params, trials))

    for i, param_value in enumerate(param_values):
        for trial in range(trials):
            X_train, X_test, y_train, y_test = train_test_split(
                dataset.data, dataset.target, test_size=test_size, stratify=dataset.target, random_state=seed + trial
            )
            y_mislabelled = noise_func(np.copy(y_train), noise_rate, seed=seed + trial)

            # Inject dynamic parameter value
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

            # Method 0: relabelling
            rf, y_corrected = train(
                forest_class, X_train, y_mislabelled,
                **train_kwargs
            )

            y_pred = rf.predict(X_test)

            accuracies_all[0, i, trial] = metrics.accuracy_score(y_test, y_pred)
            relabelling_f1_all[0, i, trial], relabelling_acc_all[0, i, trial] = relabelling_f1(y_train, y_mislabelled, y_corrected)

    for trial in range(trials):
        # Method 1: control
        if control:
            rf = forest_class(n_estimators=n_estimators * iterations, random_state = seed + trial)
            rf.fit(X_train, y_mislabelled)
            y_pred = rf.predict(X_test)
            accuracies_all[1, :, trial] = metrics.accuracy_score(y_test, y_pred)

    return accuracies_all, relabelling_f1_all, relabelling_acc_all, np.array(param_values)


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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates the number of times thresholding finds no samples over a specific training parameter at a fixed noise level.

    Args:
        param_name: Name of the parameter to vary ("K", "L", "initial_certainty", etc.)
        param_values: List of values to try for that parameter
        noise_rate: The fixed noise rate to test against

    Returns:
        accuracies_all: shape (2, len(param_values), trials)
        relabelling_f1_all: shape (1, len(param_values), trials)
        relabelling_acc_all: shape (1, len(param_values), trials)
        param_values: values that were tested
    """

    n_params = len(param_values)
    no_samples_count_all = np.zeros((1, n_params, trials))
    accuracies_all = np.zeros((1, n_params, trials))
    relabelling_f1_all = np.zeros((1, n_params, trials))
    relabelling_acc_all = np.zeros((1, n_params, trials))

    for i, param_value in enumerate(param_values):
        for trial in range(trials):
            X_train, X_test, y_train, y_test = train_test_split(
                dataset.data, dataset.target, test_size=test_size, stratify=dataset.target, random_state=seed + trial
            )
            y_mislabelled = noise_func(np.copy(y_train), noise_rate, seed=seed + trial)

            # Inject dynamic parameter value
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

            # Method 0: relabelling
            rf, y_corrected, no_samples_count = Testing_ARFs.train_log_no_threshold_samples(
                forest_class, X_train, y_mislabelled,
                **train_kwargs
            )

            y_pred = rf.predict(X_test)

            no_samples_count_all[0, i, trial] = no_samples_count
            accuracies_all[0, i, trial] = metrics.accuracy_score(y_test, y_pred)
            relabelling_f1_all[0, i, trial], relabelling_acc_all[0, i, trial] = relabelling_f1(y_train, y_mislabelled, y_corrected)

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
    Evaluates performance and relabelling success over training iterations
    at a fixed noise level.

    Returns:
        accuracies_all:      (1, iterations, trials)
        relabelling_f1_all:  (1, iterations, trials)
        relabelling_acc_all: (1, iterations, trials)
        iteration_range:     List of iteration indices (1-based)
    """

    accuracies_all = np.zeros((1, iterations, trials))
    relabelling_f1_all = np.zeros((1, iterations, trials))
    relabelling_acc_all = np.zeros((1, iterations, trials))

    for trial in range(trials):
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.data, dataset.target,
            test_size=test_size,
            stratify=dataset.target,
            random_state=seed + trial
        )
        y_mislabelled = noise_func(np.copy(y_train), noise_rate, seed=seed + trial)

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

        for i, rf in enumerate(rfs):
            y_pred = rf.predict(X_test)
            accuracies_all[0, i, trial] = metrics.accuracy_score(y_test, y_pred)
            relabelling_f1_all[0, i, trial], relabelling_acc_all[0, i, trial] = relabelling_f1(
                y_train, y_mislabelled, ys_corrected[i]
            )

    return accuracies_all, relabelling_f1_all, relabelling_acc_all, list(range(1, iterations + 1))


def relabelling_f1(y_true, y_mislabelled, y_corrected):
    noise_mask = y_mislabelled != y_true  # 1 = label was noisy
    correct_mask = y_true == y_corrected # 1 = new label is correct
    change_mask = y_mislabelled != y_corrected # 1 = relabelling occurred
    relabel_correct = noise_mask & correct_mask # 1 = noisy label was corrected

    if noise_mask.sum() == 0 and change_mask.sum() == 0:
        return 1.0, 1.0  # perfect (no noise, nothing changed)
    elif noise_mask.sum() == 0:
        return 0.0, 0.0  # nothing to fix, but changes happened = bad
    else:
        return metrics.f1_score(noise_mask, change_mask), sum(relabel_correct)/sum(noise_mask)

def process_experiment_result(results: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes experiment outputs to compute mean and standard error.

    Args:
        results: (n_methods, resolution, trials) array of results scores.

    Returns:
        mean:     (n_methods, resolution)
        se:       (n_methods, resolution)
    """

    results = np.array(results)
    return results.mean(axis=2), sem(results, axis=2)
    
def plot_with_error_band(x, y_mean, y_se, label, color, linestyle="-", alpha=0.2):
    """
    Plots a line with a shaded error band using mean and standard error.

    Parameters:
    - x: 1D array-like, x-axis values
    - y_mean: 1D array-like, mean values
    - y_se: 1D array-like, standard error values
    - label: str, label for the plot
    - color: str, color of the line and fill
    - linestyle: str, line style ('-', ':', etc.)
    - alpha: float, transparency of the shaded area
    """
    import matplotlib.pyplot as plt

    plt.plot(x, y_mean, label=label, color=color, linestyle=linestyle)
    plt.fill_between(
        x,
        y_mean - y_se,
        y_mean + y_se,
        alpha=alpha,
        color=color
    )

def plot_heatmap(matrix, x_axis, y_axis, title, label, x_title, y_title):
    iterations_range = x_axis
    n_estimators_range = y_axis

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(matrix, interpolation='nearest', cmap='viridis', origin='lower')

    ax.set_xticks(np.arange(len(n_estimators_range)))
    ax.set_yticks(np.arange(len(iterations_range)))
    ax.set_xticklabels(n_estimators_range)
    ax.set_yticklabels(iterations_range)

    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_title(title)

    cbar = fig.colorbar(cax)
    cbar.set_label(label)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def load_gmm5 ():
    # Load the data
    data1 = np.loadtxt("../Data/gmm5test.txt")
    data2 = np.loadtxt("../Data/gmm5train.txt")

    dataset = np.vstack((data1, data2))
    np.random.seed(42)
    np.random.shuffle(dataset)

    # Split into features and labels
    X = dataset[:, :2]  # first two columns: features
    y = dataset[:, 2].astype(int)   # third column: labels

    sklearn_like = {
        'data': X,
        'target': y,
        'feature_names': ['feature_1', 'feature_2'],
        'target_names': [0, 1],
        'DESCR': 'GMM5',
    }

    return Bunch(**sklearn_like)
    