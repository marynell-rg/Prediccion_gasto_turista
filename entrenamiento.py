#Importamos las librerías a utilizar

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import dump
import numpy as np
import pandas as pd

# Cargamos archivo .csv desde el workspace
data = pd.read_csv('data_tourism_final.csv')

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
    
# Columnas a transformar con logaritmo
cols_log = ['tourism_arrivals_log', 'tourism_expenditures_log', 'gdp_log', 'unemployment_log']

# Columnas a escalar
cols_to_scale = ['tourism_arrivals_log', 'tourism_expenditures_log', 'gdp_log', 'unemployment_log']

#Separamos el DF en X e y
y = data['tourism_expenditures_mean']
X = data.drop(columns=['tourism_expenditures_mean'])

# Escalado
scaler = ColumnTransformer(
    transformers=[
        ('scale', MinMaxScaler(), cols_to_scale)
    ],
    remainder='passthrough'
)

# Creamos el pipeline
pipeline = Pipeline([
    ('log_transform', LogTransformer(cols_to_transform=cols_log)),
    ('scale_features', scaler),
    ('model', GradientBoostingRegressor(random_state=42))
])

# Cros validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

# Entrenamiento del pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Entrenamos con todo el conjunto antes de guardar
pipeline.fit(X, y)
dump(pipeline, "modelo_turismo.joblib")
print("Modelo entrenado y guardado como 'modelo_turismo.joblib'")