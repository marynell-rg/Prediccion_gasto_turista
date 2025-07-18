# Importamos las librerías a utilizar
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Cargamos archivo .csv con los datos crudos
data = pd.read_csv('data_tourism_crudo.csv')

# Separar el DF en X e y
y = data['tourism_expenditures_mean']
X = data.drop(columns=['tourism_expenditures_mean'])

# Validacion de datos
# Verificación de valores nulos

if X.isnull().sum().sum() > 0:
    raise ValueError("El dataset contiene valores nulos. Revisa los datos antes de entrenar.")

# Verificación de columnas con log
cols_log = ['tourism_arrivals', 'tourism_expenditures', 'gdp']  # las que se usan en LogTransformer
for col in cols_log:
    if (X[col] < 0).any():
        raise ValueError(f"La columna '{col}' contiene valores negativos. No se puede aplicar log1p.")

# Preprocesamiento
preprocessing = joblib.load('preprocessing.joblib')

# Creamos el pipeline con el preprocesamiento y el modelo
pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('modelo', GradientBoostingRegressor(random_state=42))
])

# Entrenar el pipeline
pipeline.fit(X, y)

# Guardadamos nuestro modelo entrenado
joblib.dump(pipeline, "modelo_turismo.joblib")
print("Modelo entrenado y guardado como 'modelo_turismo.joblib'")