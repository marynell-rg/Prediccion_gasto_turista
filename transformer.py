from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Columnas a transformar con logaritmo
cols_log = ['tourism_arrivals_log', 'tourism_expenditures_log', 'gdp_log', 'unemployment_log']

# Columnas a escalar
cols_to_scale = ['tourism_arrivals_log', 'tourism_expenditures_log', 'gdp_log', 'unemployment_log']

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
    
scaler = ColumnTransformer(
    transformers=[
        ('scale', MinMaxScaler(), cols_to_scale)
    ],
    remainder='passthrough'
)