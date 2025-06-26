#Importamos las librerías a utilizar

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


# Cargamos archivo .csv desde el workspace
data = pd.read_csv('data_tourism_final.csv')

# Columnas a transformar con logaritmo
cols_log = ['tourism_arrivals_log', 'tourism_expenditures_log', 'gdp_log', 'unemployment_log']

# Columnas a escalar
cols_to_scale = cols_log

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
    ('scale_features', scaler)
])

# Guardamos el preprocesamiento
joblib.dump(pipeline, 'preprocessing.joblib')
print("Preprocesador guardado como: 'preprocessing.joblib'")