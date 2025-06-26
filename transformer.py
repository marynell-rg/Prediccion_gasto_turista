from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

# Transformación logaritmica de las columnas
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_transform):
        self.cols_to_transform = cols_to_transform

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols_to_transform:
            X[col] = np.log1p(X[col])
        return X