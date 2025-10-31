import numpy as np
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X.values, y.values, depth=0)

    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        
        n_features = X.shape[1]
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for t in thresholds:
                left_idx = X[:, feature_idx] <= t
                right_idx = ~left_idx
                
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                
                gain = information_gain(y, y[left_idx], y[right_idx])
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_thresh = t
        return split_idx, split_thresh, best_gain

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # stopping conditions
        if (depth >= self.max_depth or 
            num_labels == 1 or 
            num_samples < self.min_samples_split):
            leaf_value = self._majority_vote(y)
            return Node(value=leaf_value)

        feature_idx, threshold, gain = self._best_split(X, y)
        if feature_idx is None:
            leaf_value = self._majority_vote(y)
            return Node(value=leaf_value)

        left_idx = X[:, feature_idx] <= threshold
        right_idx = ~left_idx

        left_child = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        return Node(feature_idx, threshold, left_child, right_child)

    def _majority_vote(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        return np.array([self._predict_sample(sample, self.root) for sample in X.values])

    def _predict_sample(self, sample, node):
        if node.value is not None:
            return node.value
        if sample[node.feature] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)

    def print_tree(self, node=None, depth=0):
        """Text-based visualization of the tree."""
        if node is None:
            node = self.root
        prefix = "  " * depth
        if node.value is not None:
            print(f"{prefix}Leaf: Predict -> {node.value}")
        else:
            print(f"{prefix}Feature[{node.feature}] <= {node.threshold:.3f}?")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)


def entropy(y):
    """Compute the entropy of a label array."""
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(y, y_left, y_right):
    """Compute Information Gain from a split."""
    H = entropy(y)
    n = len(y)
    H_left = entropy(y_left)
    H_right = entropy(y_right)
    return H - (len(y_left)/n)*H_left - (len(y_right)/n)*H_right