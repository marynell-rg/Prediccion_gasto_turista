#Importamos las librerías a utilizar

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump
from transformer import LogTransformer
from pathlib import Path

# Definimos la ruta del archivo CSV y cargamos los datos
BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / 'data' / 'data_tourism_crudo.csv'
data = pd.read_csv(csv_path)

# Columnas a transformar con logaritmo
cols_log = ['tourism_arrivals', 'tourism_expenditures', 'gdp']

# Columnas a escalar
cols_to_scale = ['tourism_arrivals', 'tourism_expenditures', 'gdp', 'inflation', 'unemployment']

#Separamos el DF en X e y
y = data['tourism_expenditures_mean']
X = data.drop(columns=['tourism_expenditures_mean'])

# Pipeline del preprocesamiento
preprocessing_pipeline = Pipeline([
    ('log_transform', LogTransformer(cols_to_transform=cols_log)),
    ('scaler', ColumnTransformer([
        ('scale', MinMaxScaler(), cols_to_scale)
    ], remainder='passthrough'))
])


# Entrenamos el preprocesamiento
X_transformed = preprocessing_pipeline.fit_transform(X)

# Guardamos el preprocesamiento
joblib.dump(preprocessing_pipeline, 'preprocessing.joblib')
print("Preprocesador guardado como: 'preprocessing.joblib'")