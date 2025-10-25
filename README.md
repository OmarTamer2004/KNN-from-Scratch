# 🧠 K-Nearest Neighbors (KNN) from Scratch

## 📘 Overview
This project demonstrates a simple implementation of the **K-Nearest Neighbors (KNN)** algorithm **from scratch** using pure Python and NumPy — without relying on machine learning libraries like `scikit-learn`.

KNN is one of the simplest yet most powerful classification algorithms.  
It classifies a new data point based on the **majority class** of its **k nearest neighbors** in the training data.

---

## 🧩 How KNN Works

1. Store all training data points.
2. For a new point:
   - Compute the **distance** (usually Euclidean) between the new point and every training point.
   - Select the **k nearest neighbors**.
   - Take a **majority vote** among their labels.
3. Assign the label with the most votes.

---

## 🧮 Mathematical Formula

The distance between two points \( x_i \) and \( x_j \) is calculated as:

\[
d(x_i, x_j) = \sqrt{ \sum_{n=1}^{N} (x_i^{(n)} - x_j^{(n)})^2 }
\]

where:
- \( x_i^{(n)} \) = nth feature of the ith sample  
- \( N \) = number of features

---

## ⚙️ Implementation Details

### Class Structure:
```python
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

