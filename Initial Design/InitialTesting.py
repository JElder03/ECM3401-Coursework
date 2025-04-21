import numpy as np
from InitialAdjustedRandomForest import train, hard_train
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats import sem
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

def run_relabelling_experiment(
    dataset,
    forest_class,
    noise_func,
    n_estimators=10,
    trials=35,
    resolution=20,
    test_size=0.25,
    methods=("standard", "control"),
    relabelling = True,
    initial_certainty = 0.95,
    K = 0.5,
    L = 0.01,
    bootstrap = True,
    relabel_bunch = 1,
    seed = 42
):
    """
    Runs relabelling experiments over multiple trials and noise levels.

    Parameters:
    - dataset: sklearn-style object with `.data` and `.target`
    - forest_class: classifier class (e.g., RandomForestClassifier)
    - noise_func: callable, applies label noise, takes (y, noise_rate)
    - n_estimators: number of estimators for standard methods
    - trials: number of trials per noise level
    - resolution: number of noise steps (from 0 to 1)
    - test_size: test set proportion
    - iterations: number of adjusted RF iterations
    - methods: tuple of method names (default assumes 3: standard, bootstrapped, control)
    - relabelling: toggles relabelling capability in the adjusted random forest

    Returns:
    - accuracies_all: np.array (n_methods, resolution, trials)
    - relabelling_all: np.array (2, resolution, trials)  [only first 2 methods apply relabelling]
    - noise_levels: np.array of noise rates
    """

    n_methods = len(methods)
    accuracies_all = np.zeros((n_methods, resolution, trials))
    relabelling_f1_all = np.zeros((2, resolution, trials))
    relabelling_acc_all = np.zeros((2, resolution, trials))

    noise_levels = np.linspace(0, 1, resolution)

    for i, noise_rate in enumerate(noise_levels):
        for trial in range(trials):
            X_train, X_test, y_train, y_test = train_test_split(
                dataset.data, dataset.target, test_size=test_size, random_state = seed + trial
            )

            y_mislabelled = noise_func(np.copy(y_train), noise_rate, seed = seed + trial)

            # Method 0: standard relabelling
            if ("standard") in methods:
                rf, corrected_y, _ = train(forest_class, X_train, y_mislabelled, np.unique(dataset.target), n_estimators=n_estimators, relabelling=relabelling, initial_certainty=initial_certainty, K = K, L = L, bootstrap=bootstrap, relabel_bunch=relabel_bunch, seed = seed)
                y_pred = rf.predict(X_test)
                relabelling_f1_all[0, i, trial], relabelling_acc_all[0, i, trial] = relabelling_f1(y_train, y_mislabelled, corrected_y)
                accuracies_all[0, i, trial] = metrics.accuracy_score(y_test, y_pred)

            # Method 1: control (no relabelling, more estimators)
            if ("control") in methods:
                rf = forest_class(n_estimators=n_estimators, criterion="entropy")
                rf.fit(X_train, y_mislabelled)
                y_pred = rf.predict(X_test)
                accuracies_all[1, i, trial] = metrics.accuracy_score(y_test, y_pred)

    return accuracies_all, relabelling_f1_all, relabelling_acc_all, noise_levels

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

def process_experiment_results(
    accuracies_all,
    relabelling_f1_all=None,
    relabelling_acc_all=None,
    resolution=20,
    test_size=0.25,
    dataset_size=None  # total number of samples before train/test split
):
    """
    Processes raw experiment outputs to compute mean & SE for accuracies and relabelling success.

    Parameters:
    - accuracies_all: array-like (n_methods, resolution, trials)
    - relabelling_all: array-like (n_methods_relabelling, resolution, trials), optional
    - resolution: number of noise levels
    - test_size: test split ratio used during training
    - dataset_size: total number of original samples (before train/test split)

    Returns:
    - accuracies_mean: (n_methods, resolution)
    - accuracies_se: (n_methods, resolution)
    - relabelling_success: (n_methods_relabelling, resolution) or None
    - relabelling_se: (n_methods_relabelling, resolution) or None
    - noise_levels: (resolution,)
    """

    accuracies_all = np.array(accuracies_all)
    accuracies_mean = accuracies_all.mean(axis=2)
    accuracies_se = sem(accuracies_all, axis=2)

    relabelling_f1_success = None
    relabelling_f1_se = None
    relabelling_acc_success = None
    relabelling_acc_se = None

    if relabelling_f1_all is not None:
        relabelling_f1_all = np.array(relabelling_f1_all)
        relabelling_f1_success = relabelling_f1_all.mean(axis=2)
        relabelling_f1_se = sem(relabelling_f1_all, axis=2)

    if relabelling_acc_all is not None:
        relabelling_acc_all = np.array(relabelling_acc_all)
        relabelling_acc_success = relabelling_acc_all.mean(axis=2)
        relabelling_acc_se = sem(relabelling_acc_all, axis=2)

    noise_levels = np.linspace(0, 1, resolution)
    return accuracies_mean, accuracies_se, relabelling_f1_success, relabelling_f1_se, relabelling_acc_success, relabelling_acc_se, noise_levels


def plot_rf_decision_boundary(rf, X, y, resolution=100, class_index=1, cmap='bwr'):
    """
    Plots the decision boundary of a trained RandomForestClassifier on 2D data.

    Parameters:
    - rf: trained RandomForestClassifier
    - X: 2D input data, shape (n_samples, 2)
    - y: labels corresponding to X
    - resolution: number of grid points along each axis
    - class_index: index of the class to show the posterior probability for (default: 1)
    - cmap: colormap to use for the probability shading
    """
    if X.shape[1] != 2:
        raise ValueError("This function only works with 2D input data.")

    # Define the bounds of the grid
    x_min, x_max = X[:, 0].min() - X[:, 0].min()*0.1, X[:, 0].max() + X[:, 0].max()*0.1
    y_min, y_max = X[:, 1].min() - X[:, 1].min()*0.1, X[:, 1].max() + X[:, 1].max()*0.1

    # Create a grid of points
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Get predicted probabilities for each point in the grid
    probs = rf.predict_proba(grid)[:, class_index]
    probs = probs.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xx, yy, probs, levels=100, cmap=cmap, alpha=0.7)
    plt.colorbar(contour, label=f'P(class={class_index})')

    # Overlay the original data points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=ListedColormap(['#1f77b4', '#ff7f0e']))
    plt.title("Random Forest Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()