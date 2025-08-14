# Importamos las librerías a utilizar
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from pathlib import Path
from src.transformer import LogTransformer

# Definimos las rutas base
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'data_tourism_crudo.csv'
PREPROCESSING_PATH = BASE_DIR / 'preprocessing' / 'preprocessing.joblib'
MODEL_PATH = BASE_DIR / 'model' / 'modelo_turismo.joblib'

# Cargamos los datos
data = pd.read_csv(DATA_PATH)

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
preprocessing = joblib.load(PREPROCESSING_PATH)

# Creamos el pipeline con el preprocesamiento y el modelo
pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('modelo', GradientBoostingRegressor(random_state=42))
])

# Entrenar el pipeline
pipeline.fit(X, y)

# Guardamos nuestro modelo entrenado
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print(f"Modelo entrenado y guardado en: {MODEL_PATH}")