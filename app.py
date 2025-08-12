import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Predicción Gasto Turístico", layout="centered")

# --- Título ---
st.title("Predicción del Gasto Medio por Turista")
st.markdown("Esta aplicación estima el **Gasto medio por Turista** a partir de datos económicos y el país de destino.")

# --- Cargar código de países ---
@st.cache_data
def cargar_codigos_paises():
    df = pd.read_csv("data/country_codes_num.csv")
    return dict(zip(df["country"], df["code_num"]))

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_turismo.joblib")

# --- Inicializar ---
codigos_paises = cargar_codigos_paises()
modelo = cargar_modelo()

# --- Inputs del usuario ---
st.header("Introduce los datos:")
tourism_expenditures = st.number_input("Gasto total turístico (USD millones)", min_value=0.15, value=0.15, step=1.0)
tourism_arrivals = st.number_input("Número de turistas", min_value=900.0, value=900.0, step=1000.0)
pib = st.number_input("PIB del país (USD millones)", min_value=1.0, value=1.0, step=1000000.0)
desempleo = st.number_input("Desempleo (%)", value=0.039, step=0.1)
inflacion = st.number_input("Inflación (%)", min_value=-17.0, value=0.0, step=1.0)
pais = st.selectbox("País", list(codigos_paises.keys()))
anno = st.slider("Año", min_value=2025, max_value=2035, value=2025)

# --- Convertir datos ---
codigo_pais = codigos_paises[pais]

# Crear input DataFrame
input_data = pd.DataFrame([{
            'code_num': codigo_pais,
            'year': anno,
            'tourism_arrivals': tourism_arrivals,
            'tourism_expenditures': tourism_expenditures,
            'gdp': pib,
            'inflation': inflacion,
            'unemployment': desempleo
        }])

 # Hacer predicción
if st.button("Predecir gasto medio"):
    try:
        prediccion = (modelo.predict(input_data)[0])*(10**5)
        st.success(f"Gasto medio estimado por turista: **${prediccion:,.2f} USD**")

    except Exception as e:
        st.error(f"Error al predecir: {e}")
        st.write("📦 Datos enviados al modelo:", input_data)