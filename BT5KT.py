from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import math

# Bước 1: Tải dữ liệu Iris và chia tập dữ liệu
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bước 2: Triển khai CART với chỉ số Gini
cart_model = DecisionTreeClassifier(criterion='gini', random_state=42)
cart_model.fit(X_train, y_train)
y_pred_cart = cart_model.predict(X_test)

# Đánh giá CART
cart_accuracy = accuracy_score(y_test, y_pred_cart)
print("CART Accuracy:", cart_accuracy)
print("CART Classification Report:\n", classification_report(y_test, y_pred_cart))

# Bước 3: Triển khai ID3 (tính Information Gain)
class ID3DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * math.log2(p) for p in ps if p > 0])

    def _information_gain(self, X, y, feature_index):
        root_entropy = self._entropy(y)
        values, counts = np.unique(X[:, feature_index], return_counts=True)
        weighted_entropy = np.sum([
            (counts[i] / sum(counts)) * self._entropy(y[X[:, feature_index] == v])
            for i, v in enumerate(values)
        ])
        return root_entropy - weighted_entropy

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        for feature_index in range(X.shape[1]):
            gain = self._information_gain(X, y, feature_index)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
        return best_feature

    def _build_tree(self, X, y, depth):
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return np.bincount(y).argmax()

        feature_index = self._best_split(X, y)
        if feature_index is None:
            return np.bincount(y).argmax()

        tree = {feature_index: {}}
        for value in np.unique(X[:, feature_index]):
            subset_X = X[X[:, feature_index] == value]
            subset_y = y[X[:, feature_index] == value]
            tree[feature_index][value] = self._build_tree(subset_X, subset_y, depth + 1)
        return tree

    def _predict_sample(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        feature_index = list(tree.keys())[0]
        feature_value = x[feature_index]
        subtree = tree[feature_index].get(feature_value, None)
        if subtree is None:
            return np.bincount(y_train).argmax()
        return self._predict_sample(x, subtree)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

# Bước 4: Huấn luyện và đánh giá ID3
id3_model = ID3DecisionTreeClassifier(max_depth=5)
id3_model.fit(X_train, y_train)
y_pred_id3 = id3_model.predict(X_test)

# Đánh giá ID3
id3_accuracy = accuracy_score(y_test, y_pred_id3)
print("ID3 Accuracy:", id3_accuracy)
print("ID3 Classification Report:\n", classification_report(y_test, y_pred_id3))
