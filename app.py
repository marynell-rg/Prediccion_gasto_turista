import streamlit as st
import pandas as pd
import joblib

# --- Título principal ---
st.title("💰 Predicción del Gasto Medio por Turista")
st.markdown("Esta aplicación estima el **gasto medio por turista** a partir de indicadores económicos y el país de destino.")

# --- Cargar códigos de países ---
@st.cache_data
def cargar_codigos_paises():
    df = pd.read_csv("country_codes_num.csv")
    return dict(zip(df["country"], df["code_num"]))

# --- Cargar el modelo ---
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_turismo.joblib")

# --- Inicialización ---
codigos_paises = cargar_codigos_paises()
modelo = cargar_modelo()

# --- Inputs del usuario ---
st.header("📊 Introduce los datos:")

gasto_total = st.number_input("Gasto total (USD)", min_value=0.0, value=1_000_000.0, step=1000.0)
n_turistas = st.number_input("Número de turistas", min_value=1, value=5000, step=100)
pib = st.number_input("PIB del país (millones USD)", min_value=0.0, value=50_000.0, step=100.0)
inflacion = st.number_input("Inflación (%)", value=5.0, step=0.1)
desempleo = st.number_input("Desempleo (%)", value=10.0, step=0.1)
pais = st.selectbox("País", list(codigos_paises.keys()))
anio = st.slider("Año", min_value=2000, max_value=2030, value=2024)

# --- Convertir país a código ---
codigo_pais = codigos_paises.get(pais, 0)

# --- Crear DataFrame para el modelo ---
input_data = pd.DataFrame({
    "gasto": [gasto_total],
    "numero de turistas": [n_turistas],
    "producto interno bruto": [pib],
    "inflación": [inflacion],
    "desempleo": [desempleo],
    "país": [codigo_pais],
    "año": [anio]
})

# --- Predicción ---
if st.button("🎯 Predecir gasto medio"):
    try:
        prediccion = modelo.predict(input_data)[0]
        st.success(f"🧾 El gasto medio estimado por turista es: **${prediccion:,.2f} USD**")
    except Exception as e:
        st.error(f"❌ Error al predecir: {e}")