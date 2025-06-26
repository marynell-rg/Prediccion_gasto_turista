# Importamos las librerías a utilizar

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import dump
from transformer import LogTransformer

# Cargamos archivo .csv con los datos limpios
data = pd.read_csv('data_tourism_final.csv')

# Separar el DF en X e y
y = data['tourism_expenditures_mean']
X = data.drop(columns=['tourism_expenditures_mean'])

# Preprocesamiento
preprocessing = joblib.load('preprocessing.joblib')

pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('modelo', GradientBoostingRegressor(random_state=42))
])

# Entrenar el pipeline
pipeline.fit(X, y)

# Guardadamos nuestro modelo entrenado
joblib.dump(pipeline, "modelo_turismo.joblib")
print("Modelo entrenado y guardado como 'modelo_turismo.joblib'")