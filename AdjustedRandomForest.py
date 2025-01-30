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

def train(ensemble: RandomForestClassifier|ExtraTreesClassifier, X: list, y: list, labels: list, n_estimators: int = 100) -> RandomForestClassifier|ExtraTreesClassifier:
    INITIAL_CERTAINTY = 0.50
    NUM_CLASSES = len(labels)
    K = 0 # quartiles of exclusion
    L = 1 # learning rate
    B = 0.02 # sigmoid amplification

    y = np.array(y)
    p = np.eye(NUM_CLASSES)[y] #OHE
    p[p == 0] = (1-INITIAL_CERTAINTY)/(NUM_CLASSES-1)
    p[p == 1] = INITIAL_CERTAINTY

    u = np.array([[inv_sigmoid(probability,B) for probability in probabilities] for probabilities in p], dtype=np.float32)

    forest = ensemble(n_estimators=1, criterion='entropy', warm_start=True)

    for i in range(n_estimators):
        print(y)
        forest.set_params(n_estimators = i + 1)

        x_train, weights = select_training_data(X, y, p, K)
        x_train = np.repeat(x_train, NUM_CLASSES, axis=0)
        y_train = np.array(list(range(NUM_CLASSES)) * len(weights))
        weights = weights.flatten()


        forest.fit(x_train, y_train, sample_weight = weights)

        e = forest.predict_proba(X)
        u = update_us(u, e, p, L)
        p = update_probabilities(u, B)
        y = update_labels(y, p)
    
    return forest, y


def select_training_data(X : npt.NDArray[np.float32], y : npt.NDArray[np.float32], p : npt.NDArray[np.float32], K : float) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    
    # Dictionary to store lists of values corresponding to each unique key
    grouped_values = defaultdict(list)

    # Group values by their corresponding key
    for label, probability in zip(y, p):
        grouped_values[label].append(probability[label])

    # Compute mean for each key
    means = {label: np.mean(probabilities) for label, probabilities in grouped_values.items()}
    standard_deviations = {label: np.std(probabilities) for label, probabilities in grouped_values.items()}

    x_train = []
    weights = []

    for i in range(len(X)):
        if p[i][y[i]] >= (means[y[i]] + K * standard_deviations[y[i]]) or np.isclose(p[i][y[i]], means[y[i]] + K * standard_deviations[y[i]], atol=1e-8):
            x_train.append(X[i])
            weights.append(p[i])

    return np.array(x_train), np.array(weights)


def update_us(u : npt.NDArray[np.float32], e : npt.NDArray[np.float32], p : npt.NDArray[np.float32], L) -> npt.NDArray[np.float32]:
    u_new = u + L * (e - p)

    return u_new


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
N_CLASSES = 30
MISLABELLING = 0.3

for _ in range(1):
    Xs, ys = make_classification(n_samples=3000, n_features=50, n_redundant=0, n_informative=45, n_clusters_per_class=1, n_classes=N_CLASSES, flip_y=0.00000001)
    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=0.25)
    
    y_mislabelled = np.copy(y_train)

    for i in range(int(len(y_mislabelled) * MISLABELLING)):
        y_mislabelled[i] = (y_mislabelled[i] + random.randint(1, N_CLASSES - 1)) % N_CLASSES

    rf, corrected_y = train(RandomForestClassifier, X_train, y_mislabelled, np.unique(ys), 10)

    y_test_pred = rf.predict(X_test)

    print(sum(i!=j for i, j in zip(y_mislabelled, y_train)))
    print(sum(i!=j for i, j in zip(corrected_y, y_train)))

    scores_my.append(metrics.accuracy_score(y_test, y_test_pred))

'''
    rf = RandomForestClassifier(n_estimators=30, criterion='entropy')
    rf.fit(X_train, y_mislabelled)
    y_test_pred = rf.predict(X_test)
    scores_std.append(metrics.accuracy_score(y_test, y_test_pred))
'''

print(f"Test\nIndividual Accuracies: {scores_my}\nAverage Accuracy: {np.mean(scores_my)}\n")
print(f"Control\nIndividual Accuracies: {scores_std}\nAverage Accuracy: {np.mean(scores_std)}")
