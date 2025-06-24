import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predicción Gasto Turístico", layout="centered")

# --- Título ---
st.title("Predicción del Gasto Medio por Turista")
st.markdown("Esta aplicación estima el **gasto medio por turista** a partir de datos económicos y el país de destino.")

# --- Cargar código de países ---
@st.cache_data
def cargar_codigos_paises():
    df = pd.read_csv("country_codes_num.csv")
    return dict(zip(df["country"], df["code_num"]))

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_turismo.joblib")

# --- Inicializar ---
codigos_paises = cargar_codigos_paises()
modelo = cargar_modelo()

# --- Inputs del usuario ---
st.header("Introduce los datos:")

tourism_expenditures = st.number_input("Gasto total turístico (USD)", min_value=0.0, value=1_000_000.0, step=1000.0)
tourism_arrivals = st.number_input("Número de turistas", min_value=1, value=5000, step=100)
pib = st.number_input("PIB del país (USD millones)", min_value=0.0, value=50_000.0, step=100.0)
desempleo = st.number_input("Desempleo (%)", value=10.0, step=0.1)
pais = st.selectbox("País", list(codigos_paises.keys()))
anno = st.slider("Año", min_value=2000, max_value=2030, value=2024)

# --- Convertir datos ---
codigo_pais = codigos_paises[pais]

if st.button("Predecir gasto medio"):
    try:
        # Aplicar logaritmos
        tourism_arrivals_log = np.log1p(tourism_arrivals)
        tourism_expenditures_log = np.log1p(tourism_expenditures)
        gdp_log = np.log1p(pib * 1_000_000)
        unemployment_log = np.log1p(desempleo)

        input_data = pd.DataFrame([{
            'code_num': codigo_pais,
            'year': anno,
            'tourism_arrivals_log': tourism_arrivals_log,
            'tourism_expenditures_log': tourism_expenditures_log,
            'gdp_log': gdp_log,
            'unemployment_log': unemployment_log
        }])

        # Hacer predicción
        prediccion_total = modelo.predict(input_data)[0]

        # Calcular gasto medio
        gasto_medio = prediccion_total / np.expm1(tourism_arrivals_log)

        st.success(f"Gasto medio estimado por turista: **${gasto_medio:,.2f} USD**")

    except Exception as e:
        st.error(f"Error al predecir: {e}")