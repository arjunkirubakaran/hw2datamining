import math
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter, defaultdict
# Do not import any other libraries. You can use built-in functions and the above imports only.


###################### Task-1 ################################################

def entropy(labels):
    """
    Compute entropy of a list of class labels.
    """
    n = len(labels)
    if n == 0:
        return 0.0

    counts = Counter(labels)
    ent = 0.0
    for c in counts:
        p = counts[c] / n
        ent -= p * math.log(p, 2)
    return ent


def information_gain(dataset):
    """
    Input:
        dataset: list of lists
                 last column is label
    Output:
        list of information gain for each feature
    """
    dataset = np.array(dataset)
    X = dataset[:, :-1]
    y = dataset[:, -1]

    base_entropy = entropy(y)
    n_samples = len(y)
    n_features = X.shape[1]

    ig_list = []

    for j in range(n_features):
        # group by feature value
        feature_values = X[:, j]
        value_groups = defaultdict(list)

        for i, v in enumerate(feature_values):
            value_groups[v].append(y[i])

        # compute conditional entropy
        cond_ent = 0.0
        for v in value_groups:
            subset = value_groups[v]
            cond_ent += (len(subset) / n_samples) * entropy(subset)

        ig = base_entropy - cond_ent
        ig_list.append(ig)

    return ig_list



###################### Task-2 ################################################
def perceptron_gradient_descent(X, y, w_init, b_init, lr=1.0, max_iter=100):
    """
    Parameters:
        X : list of feature vectors
        y : list of labels (-1 or +1)
        w_init : initial weight vector
        b_init : initial bias
        lr : learning rate
        max_iter : maximum iterations
        
    Returns:
        w, b
    """
    X = np.array(X)
    y = np.array(y)

    w = np.array(w_init, dtype=float)
    b = float(b_init)

    for _ in range(max_iter):
        misclassified = []

        # find all misclassified samples
        for i in range(len(X)):
            if y[i] * (np.dot(w, X[i]) + b) <= 0:
                misclassified.append(i)

        # stop if none
        if len(misclassified) == 0:
            break

        # deterministic rule: choose largest index
        idx = max(misclassified)

        # perceptron update
        w = w + lr * y[idx] * X[idx]
        b = b + lr * y[idx]

    return w, b
