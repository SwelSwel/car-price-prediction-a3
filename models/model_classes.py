import numpy as np
import time
import matplotlib.pyplot as plt


class NoPenalty:
    """No regularization penalty"""
    def __call__(self, theta):
        return 0
    def derivation(self, theta):
        return np.zeros(theta.shape)


class RidgePenalty:
    """L2 regularization penalty, copied from Regularization.ipynb"""
    def __init__(self, l):
        self.l = l
    def __call__(self, theta):
        return self.l * np.sum(theta ** 2)       # λΣθ²
    def derivation(self, theta):
        return self.l * 2 * theta                 # 2λθ


class LogisticRegression:
    """Multinomial Logistic Regression with optional Ridge penalty.
    Based on 02 - Multinomial Logistic Regression.ipynb"""

    def __init__(self, k, n, method, alpha=0.001, max_iter=5000, regularization=NoPenalty()):
        self.k = k              # number of classes
        self.n = n              # number of features (incl. intercept)
        self.alpha = alpha      # learning rate
        self.max_iter = max_iter
        self.method = method    # "batch", "minibatch", or "sto"
        self.regularization = regularization

    def fit(self, X, Y):
        self.W = np.random.rand(self.n, self.k)  # W shape: (n, k)
        self.losses = []

        if self.method == "batch":
            # Use all samples every iteration
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad = self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}: {loss:.4f}")
            print(f"Time taken: {time.time() - start_time:.2f}s")

        elif self.method == "minibatch":
            # Use 30% of samples per iteration
            start_time = time.time()
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0])
                batch_X = X[ix:ix+batch_size]
                batch_Y = Y[ix:ix+batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}: {loss:.4f}")
            print(f"Time taken: {time.time() - start_time:.2f}s")

        elif self.method == "sto":
            # Use one sample per iteration
            start_time = time.time()
            list_of_used_ix = []
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while i in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx]
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                list_of_used_ix.append(i)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = []
                if i % 500 == 0:
                    print(f"Loss at iteration {i}: {loss:.4f}")
            print(f"Time taken: {time.time() - start_time:.2f}s")
        else:
            raise ValueError('Method must be: "batch", "minibatch" or "sto".')

    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        loss = -np.sum(Y * np.log(h + 1e-15)) / m + self.regularization(self.W)
        error = h - Y
        grad = self.softmax_grad(X, error) + self.regularization.derivation(self.W)
        return loss, grad

    def softmax(self, z):
        # Subtract row max for numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return X.T @ error

    def h_theta(self, X, W):
        # Prediction: softmax(X @ W), output shape (m, k)
        return self.softmax(X @ W)

    def predict(self, X_test):
        # Return class with highest probability
        return np.argmax(self.h_theta(X_test, self.W), axis=1)

    def plot(self):
        plt.plot(np.arange(len(self.losses)), self.losses, label="Train Losses")
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

    # --- Classification metrics from scratch ---

    def accuracy(self, y, yhat):
        # correct predictions / all predictions
        return np.sum(y == yhat) / len(y)

    def precision(self, y, yhat, c):
        # TP / (TP + FP) for class c
        TP = np.sum((yhat == c) & (y == c))
        FP = np.sum((yhat == c) & (y != c))
        if TP + FP == 0:
            return 0.0
        return TP / (TP + FP)

    def recall(self, y, yhat, c):
        # TP / (TP + FN) for class c
        TP = np.sum((yhat == c) & (y == c))
        FN = np.sum((yhat != c) & (y == c))
        if TP + FN == 0:
            return 0.0
        return TP / (TP + FN)

    def f1_score(self, y, yhat, c):
        # 2 * precision * recall / (precision + recall) for class c
        p = self.precision(y, yhat, c)
        r = self.recall(y, yhat, c)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def macro_precision(self, y, yhat):
        # Mean precision across all classes
        return np.mean([self.precision(y, yhat, c) for c in range(self.k)])

    def macro_recall(self, y, yhat):
        return np.mean([self.recall(y, yhat, c) for c in range(self.k)])

    def macro_f1(self, y, yhat):
        return np.mean([self.f1_score(y, yhat, c) for c in range(self.k)])

    def weighted_precision(self, y, yhat):
        # Precision weighted by class support
        total = len(y)
        return np.sum([(np.sum(y == c) / total) * self.precision(y, yhat, c) for c in range(self.k)])

    def weighted_recall(self, y, yhat):
        total = len(y)
        return np.sum([(np.sum(y == c) / total) * self.recall(y, yhat, c) for c in range(self.k)])

    def weighted_f1(self, y, yhat):
        total = len(y)
        return np.sum([(np.sum(y == c) / total) * self.f1_score(y, yhat, c) for c in range(self.k)])