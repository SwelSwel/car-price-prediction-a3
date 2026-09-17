import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class LinearRegression(object):
    kfold = KFold(n_splits=3)
    
    def __init__(self, regularization, lr=0.001, num_epochs=500, method='batch', batch_size=50, cv=kfold, init_method='zeros', momentum=0.0):
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

class NoPenalty:
    def __call__(self, theta):
        return 0
    def derivation(self, theta):
        return np.zeros(theta.shape)

class Normal(LinearRegression):
    def __init__(self, method='batch', lr=0.001, num_epochs=500, batch_size=50, cv=LinearRegression.kfold, init_method='zeros', momentum=0.0):
        self.regularization = NoPenalty()
        super().__init__(self.regularization, lr=lr, method=method, num_epochs=num_epochs, batch_size=batch_size, cv=cv, init_method=init_method, momentum=momentum)