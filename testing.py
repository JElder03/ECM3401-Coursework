import numpy as np
from AdjustedRandomForest import train
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
    iterations=10,
    methods=("standard", "bootstrapped", "control"),
    relabelling = True,
    initial_certainty = 0.95
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
    relabelling_all = np.zeros((2, resolution, trials))

    noise_levels = np.linspace(0, 1, resolution)

    for i, noise_rate in enumerate(noise_levels):
        for trial in range(trials):
            X_train, X_test, y_train, y_test = train_test_split(
                dataset.data, dataset.target, test_size=test_size
            )
            y_mislabelled = noise_func(np.copy(y_train), noise_rate)

            # Method 0: standard relabelling
            if ("standard") in methods:
                rf, corrected_y = train(forest_class, X_train, y_mislabelled, np.unique(dataset.target), n_estimators=n_estimators, iterations=iterations, relabelling=relabelling, initial_certainty=initial_certainty)
                y_pred = rf.predict(X_test)
                relabelling_all[0, i, trial] = np.sum(corrected_y != y_train)
                accuracies_all[0, i, trial] = metrics.accuracy_score(y_test, y_pred)

            # Method 1: bootstrapped relabelling
            if ("bootstrapped") in methods:
                rf, corrected_y = train(forest_class, X_train, y_mislabelled, np.unique(dataset.target), n_estimators=n_estimators, bootstrapping=True, iterations=iterations, relabelling=relabelling, initial_certainty=initial_certainty)
                y_pred = rf.predict(X_test)
                relabelling_all[1, i, trial] = np.sum(corrected_y != y_train)
                accuracies_all[1, i, trial] = metrics.accuracy_score(y_test, y_pred)

            # Method 2: control (no relabelling, more estimators)
            if ("control") in methods:
                rf = forest_class(n_estimators=n_estimators * iterations, criterion="entropy")
                rf.fit(X_train, y_mislabelled)
                y_pred = rf.predict(X_test)
                accuracies_all[2, i, trial] = metrics.accuracy_score(y_test, y_pred)

    return accuracies_all, relabelling_all, noise_levels

def process_experiment_results(
    accuracies_all,
    relabelling_all=None,
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

    relabelling_success = None
    relabelling_se = None

    if relabelling_all is not None:
        relabelling_all = np.array(relabelling_all)
        n_relabel_methods = relabelling_all.shape[0]
        relabelling_success = np.ones((n_relabel_methods, resolution))
        relabelling_se = np.zeros((n_relabel_methods, resolution))

        if dataset_size is None:
            raise ValueError("dataset_size must be provided when relabelling_all is used")

        n_samples = dataset_size * (1 - test_size)
        noise_levels = np.linspace(0, 1, resolution)

        for i in range(1, resolution):  # skip 0 to avoid division by zero
            for j in range(n_relabel_methods):
                denom = n_samples * noise_levels[i]
                if denom == 0:
                    continue
                rates = relabelling_all[j][i] / denom
                relabelling_success[j][i] = 1 - np.mean(rates)
                relabelling_se[j][i] = sem(1 - rates)
    else:
        noise_levels = np.linspace(0, 1, resolution)

    return accuracies_mean, accuracies_se, relabelling_success, relabelling_se, noise_levels

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