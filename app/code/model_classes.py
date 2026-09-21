import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold




class NoPenalty:
    def __call__(self, theta):
        return 0
    def derivation(self, theta):
        return np.zeros(theta.shape)


class RidgePenalty:
    """L2 regularization penalty"""
    def __init__(self, l):
        self.l = l
    def __call__(self, theta):
        return self.l * np.sum(theta ** 2)
    def derivation(self, theta):
        return self.l * 2 * theta


# ===== A2: Linear Regression =====

class LinearRegression(object):
    kfold = KFold(n_splits=3)

    def __init__(self, regularization, lr=0.001, num_epochs=500, method='batch',
                 batch_size=50, cv=kfold, init_method='zeros', momentum=0.0):
        self.regularization = regularization
        self.lr = lr
        self.num_epochs = num_epochs
        self.method = method
        self.batch_size = batch_size
        self.cv = cv
        self.init_method = init_method
        self.momentum = momentum

    def predict(self, X):
        return X @ self.theta

    def _coef(self):
        return self.theta[1:]

    def _bias(self):
        return self.theta[0]


class Normal(LinearRegression):
    def __init__(self, method='batch', lr=0.001, num_epochs=500, batch_size=50,
                 cv=LinearRegression.kfold, init_method='zeros', momentum=0.0):
        self.regularization = NoPenalty()
        super().__init__(self.regularization, lr=lr, method=method,
                         num_epochs=num_epochs, batch_size=batch_size,
                         cv=cv, init_method=init_method, momentum=momentum)


#A3: Logistic Regression

class LogisticRegression:
    """Multinomial Logistic Regression with optional Ridge penalty.
    Based on 02 - Multinomial Logistic Regression.ipynb"""

    def __init__(self, k, n, method, alpha=0.001, max_iter=5000, regularization=NoPenalty()):
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        self.regularization = regularization

    def fit(self, X, Y):
        self.W = np.random.rand(self.n, self.k)
        self.losses = []

        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad = self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}: {loss:.4f}")
            print(f"Time taken: {time.time() - start_time:.2f}s")

        elif self.method == "minibatch":
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
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return X.T @ error

    def h_theta(self, X, W):
        return self.softmax(X @ W)

    def predict(self, X_test):
        return np.argmax(self.h_theta(X_test, self.W), axis=1)

    def plot(self):
        plt.plot(np.arange(len(self.losses)), self.losses, label="Train Losses")
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

    def accuracy(self, y, yhat):
        return np.sum(y == yhat) / len(y)

    def precision(self, y, yhat, c):
        TP = np.sum((yhat == c) & (y == c))
        FP = np.sum((yhat == c) & (y != c))
        if TP + FP == 0:
            return 0.0
        return TP / (TP + FP)

    def recall(self, y, yhat, c):
        TP = np.sum((yhat == c) & (y == c))
        FN = np.sum((yhat != c) & (y == c))
        if TP + FN == 0:
            return 0.0
        return TP / (TP + FN)

    def f1_score(self, y, yhat, c):
        p = self.precision(y, yhat, c)
        r = self.recall(y, yhat, c)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def macro_precision(self, y, yhat):
        return np.mean([self.precision(y, yhat, c) for c in range(self.k)])

    def macro_recall(self, y, yhat):
        return np.mean([self.recall(y, yhat, c) for c in range(self.k)])

    def macro_f1(self, y, yhat):
        return np.mean([self.f1_score(y, yhat, c) for c in range(self.k)])

    def weighted_precision(self, y, yhat):
        total = len(y)
        return np.sum([(np.sum(y == c) / total) * self.precision(y, yhat, c) for c in range(self.k)])

    def weighted_recall(self, y, yhat):
        total = len(y)
        return np.sum([(np.sum(y == c) / total) * self.recall(y, yhat, c) for c in range(self.k)])

    def weighted_f1(self, y, yhat):
        total = len(y)
        return np.sum([(np.sum(y == c) / total) * self.f1_score(y, yhat, c) for c in range(self.k)])