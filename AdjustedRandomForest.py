import math
from collections import defaultdict
import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.datasets import make_classification
import random
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sys

#seed = random.randrange(sys.maxsize)
#rng = random.Random(2844208129536504762)
#print("Seed was:", seed)
random.seed(1)
np.random.seed(1)

def train(ensemble: RandomForestClassifier|ExtraTreesClassifier, X: list, y: list, labels: list, n_estimators: int = 100) -> RandomForestClassifier|ExtraTreesClassifier:
    INITIAL_CERTAINTY = 0.95
    NUM_CLASSES = len(labels)
    K = 0.1 # quartiles of exclusion
    L = 0.03 # learning rate
    B = 0.02 # sigmoid amplification

    y = np.array(y)
    p = np.eye(NUM_CLASSES)[y] #OHE
    p[p == 0] = (1-INITIAL_CERTAINTY)/(NUM_CLASSES-1)
    p[p == 1] = INITIAL_CERTAINTY

    u = np.array([[inv_sigmoid(probability,B) for probability in probabilities] for probabilities in p], dtype=np.float32)

    forest = ensemble(n_estimators=1, criterion='entropy', warm_start=True, max_depth = 2, random_state=2)
    x_train = X

    weights = p

    #IQR?

    for i in range(n_estimators):
        forest.set_params(n_estimators = i + 1)

        x_train = np.repeat(x_train, NUM_CLASSES, axis=0)
        y_train = np.array(list(range(NUM_CLASSES)) * len(weights))
        weights = weights.flatten()

        forest.fit(x_train, y_train, sample_weight = weights)

        print(p[3])
        e = forest.predict_proba(X)
        print(e[3])
        u = update_us(u, e, p, L)
        p = update_probabilities(u, B)
        y = update_labels(y, p)

        selection = select_training_data(X, y, e, K)
        x_train = [X[i] for i in selection]
        weights = np.array([p[i] for i in selection])

    print(p[3])
    return forest, y

def select_training_data(X : npt.NDArray[np.float32], y : npt.NDArray[np.float32], e : npt.NDArray[np.float32], K : float) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    
    # Dictionary to store lists of values corresponding to each unique label
    grouped_values = defaultdict(list)

    # Group values by their corresponding label
    for label, probability in zip(y, e):
        grouped_values[label].append(probability[label])

    # Compute mean for each label
    means = {label: np.mean(probabilities) for label, probabilities in grouped_values.items()}
    standard_deviations = {label: np.std(probabilities) for label, probabilities in grouped_values.items()}


    selection = []

    for i in range(len(X)):
        #if means[y[i]] + K * standard_deviations[y[i]] <
        if e[i][y[i]] >= (means[y[i]] + K * standard_deviations[y[i]]) or np.isclose(e[i][y[i]], means[y[i]] + K * standard_deviations[y[i]], atol=1e-8):
            selection.append(i)

    return np.array(selection)


def update_us(u : npt.NDArray[np.float32], e : npt.NDArray[np.float32], p : npt.NDArray[np.float32], L) -> npt.NDArray[np.float32]:
    return u + L * (e - p)


def update_probabilities(u : npt.NDArray[np.float32], B : float):
    p = np.array([[sigmoid(probability,B) for probability in probabilities] for probabilities in u], dtype=np.float32) # calculate p from u (sigmoid)

    for i in range(len(p)):  # Iterate over rows
        min_value = np.min(p[i])

        if min_value < 0:
            shifted = p[i] - min_value + 1e-8  # Shift values
        else:
            shifted = p[i]

        total = np.sum(shifted)
        p[i] = shifted / total  # Normalize in-place
        
        if not np.isclose(np.sum(p[i]), 1, atol=1e-6):  # Check sum stability
            raise Exception(f"Normalization failed for row {i}: {np.sum(p[i])}")
        
    return p


def update_labels(y : npt.NDArray[np.float32], p : npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    for i in range(len(y)):
        y[i] = np.argmax(p[i] + 1)
    return y


def sigmoid(x, b):
    return (1+math.tanh(x/b))/2


def inv_sigmoid(x, b):
    return math.atanh(2 * x - 1) * b


# MISLABELLING
scores_my = []
scores_std = []
N_CLASSES = 2
MISLABELLING = 0.05

for _ in range(1):
    Xs, ys = make_classification(n_samples=100, n_features=5, n_redundant=0, n_informative=3, n_clusters_per_class=1, n_classes=N_CLASSES, flip_y=0, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.25)
    
    y_mislabelled = np.copy(y_train)

    # for i in range(int(len(y_mislabelled) * MISLABELLING)):
    #    y_mislabelled[i] = (y_mislabelled[i] + random.randint(1, N_CLASSES - 1)) % N_CLASSES
    y_mislabelled[0] = (y_mislabelled[0] + 1) % N_CLASSES

    rf, corrected_y = train(RandomForestClassifier, X_train, y_mislabelled, np.unique(ys), 5)

    y_test_pred = rf.predict(X_test)

    print(sum(i!=j for i, j in zip(y_mislabelled, y_train)))
    print(sum(i!=j for i, j in zip(corrected_y, y_train)))
    print([i!=j for i, j in zip(corrected_y, y_train)])

    scores_my.append(metrics.accuracy_score(y_test, y_test_pred))


    rf = RandomForestClassifier(n_estimators=30, criterion='entropy')
    rf.fit(X_train, y_mislabelled)
    y_test_pred = rf.predict(X_test)
    scores_std.append(metrics.accuracy_score(y_test, y_test_pred))


print(f"Test\nIndividual Accuracies: {scores_my}\nAverage Accuracy: {np.mean(scores_my)}\n")
print(f"Control\nIndividual Accuracies: {scores_std}\nAverage Accuracy: {np.mean(scores_std)}")
