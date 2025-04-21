import math
from collections import defaultdict
import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

def train(ensemble: RandomForestClassifier|ExtraTreesClassifier, X: list, y: list, labels: list, n_estimators: int = 100, initial_certainty = 0.95, K = 0.5, L = 0.01, B = 0.02, relabelling = True, bootstrap = True, relabel_bunch = 1, seed = None) -> RandomForestClassifier|ExtraTreesClassifier:
    NUM_CLASSES = len(labels)

    y = np.array(y)
    p = np.eye(NUM_CLASSES)[y] #OHE
    p[p == 0] = (1-initial_certainty)/(NUM_CLASSES-1)
    p[p == 1] = initial_certainty

    u = np.array([[inv_sigmoid(probability,B) for probability in probabilities] for probabilities in p], dtype=np.float32)
    forest = ensemble(n_estimators=1, criterion='entropy', warm_start=True, max_depth = 3, bootstrap = bootstrap, random_state = seed)
    
    x_train = np.copy(X)
    weights = np.copy(p)
    selection_size = []
    #IQR?

    for i in range(n_estimators):
        forest.set_params(n_estimators = i + 1)

        x_train = np.repeat(x_train, NUM_CLASSES, axis=0)
        y_train = np.array(list(range(NUM_CLASSES)) * len(weights))
        weights = weights.flatten()

        forest.fit(x_train, y_train, sample_weight = weights)

        if relabelling and i % relabel_bunch == relabel_bunch - 1:
            e = forest.estimators_[-1].predict_proba(X)
            for j in range(2, relabel_bunch + 1):
                e += forest.estimators_[-j].predict_proba(X)
            e /= relabel_bunch
            u = update_us(u, e, p, L)
            p = update_probabilities(u, B)
            y = update_labels(y, p)

        e = forest.estimators_[-1].predict_proba(X)
        selection, relabelling_selection = select_training_data(X, y, e, K)
        x_train = [X[i] for i in selection]
        weights = np.array([p[i] for i in selection])
        
        selection_size.append(len(selection))
    return forest, y, p

def select_training_data(X : npt.NDArray[np.float32], y : npt.NDArray[np.float32], e : npt.NDArray[np.float32], K : float) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    
    # Group values by their label
    grouped_values = defaultdict(list)

    for label, probability in zip(y, e):
        grouped_values[label].append(probability[label])

    # Compute mean and std for each label
    means = {label: np.mean(probabilities) for label, probabilities in grouped_values.items()}
    standard_deviations = {label: np.std(probabilities) for label, probabilities in grouped_values.items()}

    selection = []
    relabelling_selection = []

    for i in range(len(X)):
        if e[i][y[i]] >= (means[y[i]] + K * standard_deviations[y[i]]):
            selection.append(i)
        elif e[i][y[i]] >= (means[y[i]] - K * standard_deviations[y[i]]):
            relabelling_selection.append(i)
    
    # Use all samples if none are acceptable
    if not selection:
        selection = list(range(len(X)))

    return (np.array(selection), np.array(relabelling_selection))


def update_us(u : npt.NDArray[np.float32], e : npt.NDArray[np.float32], p : npt.NDArray[np.float32], L) -> npt.NDArray[np.float32]:
    return u + L * (e - p)


def update_probabilities(u : npt.NDArray[np.float32], B : float):
    p = np.array([[sigmoid(probability,B) for probability in probabilities] for probabilities in u]) # calculate p from u (sigmoid)

    # Normalize
    for i in range(len(p)):
        total = np.sum(p[i])
        p[i] =  p[i] / total
    return p


def update_labels(y : npt.NDArray[np.float32], p : npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    for i in range(len(y)):
        y[i] = np.argmax(p[i])
    return y


def sigmoid(x, b):
    return (1+math.tanh(x/b))/2


def inv_sigmoid(x, b):
    return math.atanh(2 * x - 1) * b

'''
e = forest.predict_proba(X)
        selection, relabelling_selection = select_training_data(X, y, e, K)
        x_train = [X[i] for i in selection]
        weights = np.array([p[i] for i in selection])

        e = forest.estimators_[-1].predict_proba(X)

        for i in relabelling_selection:
            u[i] = update_us(u[i], e[i], p[i], L)[0]
            p[i] = update_probabilities([u[i]], B)[0]
            y[i] = update_labels([y[i]], [p[i]])[0]
        
        if len(selection) == len(X):
            all_excluded += 1
'''

def hard_train(ensemble: RandomForestClassifier|ExtraTreesClassifier, X: list, y: list, labels: list, n_estimators: int = 100, initial_certainty = 0.95, K = 0.5, L = 0.01, B = 0.02, relabelling = True, bootstrap = True) -> RandomForestClassifier|ExtraTreesClassifier:
    NUM_CLASSES = len(labels)

    y = np.array(y)
    p = np.eye(NUM_CLASSES)[y] #OHE
    p[p == 0] = (1-initial_certainty)/(NUM_CLASSES-1)
    p[p == 1] = initial_certainty

    u = np.array([[inv_sigmoid(probability,B) for probability in probabilities] for probabilities in p], dtype=np.float32)

    forest = ensemble(n_estimators=1, criterion='entropy', warm_start=True, bootstrap = bootstrap)
    
    x_train = np.copy(X)

    weights = np.copy(p)
    selection_size = []
    #IQR?

    for i in range(n_estimators):
        forest.set_params(n_estimators = i + 1)

        forest.fit(x_train, y)
        
        e = forest.estimators_[-1].predict_proba(X)
        if relabelling:
            u = update_us(u, e, p, L)
            p = update_probabilities(u, B)
            y = update_labels(y, p)

        e = forest.predict_proba(X)
        selection, relabelling_selection = select_training_data(X, y, e, K)
        x_train = [X[i] for i in selection]
        weights = np.array([p[i] for i in selection])
        
        selection_size.append(len(selection))

    print(f"All selection_size: {selection_size}")
    return forest, y