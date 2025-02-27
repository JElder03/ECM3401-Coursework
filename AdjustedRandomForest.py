import math
import random
from collections import defaultdict
import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.datasets import make_classification, load_wine
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics

def train(ensemble: RandomForestClassifier|ExtraTreesClassifier, X: list, y: list, labels: list, n_estimators: int = 100) -> RandomForestClassifier|ExtraTreesClassifier:
    INITIAL_CERTAINTY = 0.95
    NUM_CLASSES = len(labels)
    K = 0.5 # quartiles of exclusion
    L = 0.01 # learning rate
    B = 0.02 # sigmoid amplification

    batch_size = math.floor(len(y)/n_estimators)
    y_ordered = np.copy(y)
    X_ordered = np.copy(X)

    y = np.array(y)
    p = np.eye(NUM_CLASSES)[y] #OHE
    p[p == 0] = (1-INITIAL_CERTAINTY)/(NUM_CLASSES-1)
    p[p == 1] = INITIAL_CERTAINTY

    u = np.array([[inv_sigmoid(probability,B) for probability in probabilities] for probabilities in p], dtype=np.float32)

    forest = ensemble(n_estimators=1, criterion='entropy', bootstrap= False, warm_start=True, max_depth = 2, class_weight = {0: 1, 1: 1, 2:1})

    for i in range(n_estimators):
        X, y = shuffle(X, y)
        X_batch = X[i*batch_size:(i+1)*batch_size]
        y_batch = y[i*batch_size:(i+1)*batch_size]

        missing_labels = set(labels) - set(y_batch)

        for label in missing_labels:
            y_batch = np.append(y_batch,label)
            X_batch = np.vstack((X_batch, np.zeros((1, X_batch.shape[1]))))  # Stack along axis 0 (rows)


        class_weights = {label: 1 for label in set(labels)} | {label: 0 for label in missing_labels}
        forest.set_params(n_estimators = i + 1, class_weight = class_weights)

        forest.fit(X_batch, y_batch)
    
    for _ in range(10):
        forest_prev = forest

        forest = ensemble(n_estimators=1, criterion='entropy', bootstrap= False, warm_start=True, max_depth = 2)
        X, y = shuffle(X, y)

        for i in range (n_estimators):
            X_batch = X[i*batch_size:(i+1)*batch_size]
            y_batch = y[i*batch_size:(i+1)*batch_size]

            e = forest_prev.predict_proba(X_batch)

            selection = select_training_data(X_batch, y_batch, e, K)

            X_batch = [X_batch[i] for i in selection]
            y_batch = [y_batch[i] for i in selection]

            missing_labels = set(labels) - set(y_batch)
            for label in missing_labels:
                y_batch.append(label)
                X_batch.append([0 for _ in range(len(X_batch[0]))])
            
            class_weights = {label: 1 for label in set(labels)} | {label: 0 for label in missing_labels}

            forest.set_params(n_estimators = i + 1, class_weight = class_weights)
            forest.fit(X_batch, y_batch)

        
        e = forest.predict_proba(X_ordered)
        u = update_us(u, e, p, L)
        p = update_probabilities(u, B)
        y = update_labels(y_ordered, p)
        X = np.copy(X_ordered)
    return forest, y

def select_training_data(X : npt.NDArray[np.float32], y : npt.NDArray[np.float32], e : npt.NDArray[np.float32], K : float) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    
    # Group values by their label
    grouped_values = defaultdict(list)

    for label, probability in zip(y, e):
        grouped_values[label].append(probability[label])

    # Compute mean and std for each label
    means = {label: np.mean(probabilities) for label, probabilities in grouped_values.items()}
    standard_deviations = {label: np.std(probabilities) for label, probabilities in grouped_values.items()}

    selection = []

    for i in range(len(X)):
        if e[i][y[i]] >= (means[y[i]] + K * standard_deviations[y[i]]):
            selection.append(i)

    return np.array(selection)


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


# MISLABELLING
scores_my = []
scores_std = []
MISLABELLING = 0.60

for _ in range(100):
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target)

    y_mislabelled = np.copy(y_train)

    for i in range(int(len(y_mislabelled) * MISLABELLING)):
        y_mislabelled[i] = (y_mislabelled[0] + random.randint(1,2)) % len(np.unique(wine.target))

    rf, corrected_y = train(RandomForestClassifier, X_train, y_mislabelled, np.unique(wine.target), 5)
    
    y_test_pred = rf.predict(X_test)

    print(sum(i!=j for i, j in zip(y_mislabelled, y_train)))
    print(sum(i!=j for i, j in zip(corrected_y, y_train)))
    #print([int(i!=j) for i, j in zip(corrected_y, y_train)])
    mismatch_indices = [index for index, (i, j) in enumerate(zip(corrected_y, y_train)) if i != j]
    print(mismatch_indices)

    scores_my.append(metrics.accuracy_score(y_test, y_test_pred))


    rf = RandomForestClassifier(n_estimators=50, criterion='entropy', max_depth = 2)
    rf.fit(X_train, y_mislabelled)
    y_test_pred = rf.predict(X_test)
    scores_std.append(metrics.accuracy_score(y_test, y_test_pred))

# Is accuracy about 50% of data at no mislabelling

print(f"Test\nIndividual Accuracies: {scores_my}\nAverage Accuracy: {np.mean(scores_my)}\n")
print(f"Control\nIndividual Accuracies: {scores_std}\nAverage Accuracy: {np.mean(scores_std)}")