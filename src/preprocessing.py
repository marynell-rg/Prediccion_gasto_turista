#Importamos las librerías a utilizar

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from src.transformer import LogTransformer

# Definimos las ruta base
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'data_tourism_crudo.csv'
PREPROCESSING_PATH = BASE_DIR / 'preprocessing' / 'preprocessing.joblib'

# Cargamos datos
data = pd.read_csv(DATA_PATH)

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

# Guardamos el preprocesamiento en preprocessing/
PREPROCESSING_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(preprocessing_pipeline, PREPROCESSING_PATH)

print(f"✅ Preprocesador guardado en: {PREPROCESSING_PATH}")