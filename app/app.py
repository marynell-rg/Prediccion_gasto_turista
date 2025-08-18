import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR)) 

from src.transformer import LogTransformer

# --- Aplicación Streamlit ---
st.set_page_config(page_title="Predicción Gasto Turístico", layout="centered")

# --- Barra lateral ---
st.sidebar.title("Acerca de esta App")
st.sidebar.markdown("""
Esta aplicación permite **predecir el gasto medio por turista** a partir de datos económicos del país y del número de turistas.  

**Cómo usarla:**  
1. Introduce los valores económicos y turísticos en los campos de entrada.  
2. Selecciona el país y el año.  
3. Haz clic en el botón **Predecir gasto medio**.  
4. Obtendrás la estimación del gasto medio por turista en USD.  

**Notas:**  
- Asegúrate de ingresar valores válidos para cada campo.  
- La predicción está basada en un modelo entrenado con datos históricos.

**Unidades:**
- Gasto total turístico: USD millones
- Número de turistas: número de personas
- PIB del país: USD millones
- Desempleo: porcentaje
- Inflación: porcentaje
""")

# --- Imagen profesional ---
st.image("assets/turismo.jpg", caption="Turismo y economía", width=700)

# --- Título ---
st.title("Predicción del Gasto Medio por Turista")
st.markdown("Esta aplicación estima el **Gasto medio por Turista** a partir de datos económicos y el país de destino.")

# Desde donde esta leyendo los datos
print("BASE_DIR:", BASE_DIR)
print("CSV path:", BASE_DIR / "data" / "country_codes_num.csv")

# --- Cargar código de países ---
@st.cache_data
def cargar_codigos_paises():
    csv_path = BASE_DIR / "data" / "country_codes_num.csv"
    if not csv_path.exists():
        st.error(f"No se encontró el archivo: {csv_path}")
        st.stop()
    df = pd.read_csv(csv_path)
    return dict(zip(df["country"], df["code_num"]))

# --- Cargar Modelo ---
@st.cache_resource
def cargar_modelo():
    model_path = BASE_DIR / "model" / "modelo_turismo.joblib"
    if not model_path.exists():
        st.error(f"No se encontró el archivo: {model_path}")
        st.stop()
    return joblib.load(model_path)

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
anno = st.selectbox("Año", [2025, 2026, 2027, 2028, 2029, 2030, 2031])

# --- Convertir datos ---
codigo_pais = codigos_paises[pais]

# Crear input DataFrame
input_data = pd.DataFrame([{
            'code_num': codigo_pais,
            'year': 2025,  # Año actual
            'tourism_arrivals': tourism_arrivals,
            'tourism_expenditures': tourism_expenditures,
            'gdp': pib,
            'inflation': inflacion,
            'unemployment': desempleo
        }])

# Hacer predicción
if st.button("Predecir gasto medio (2025-2031)"):
    try:
        # Generar dataframe para próximos 5 años
        futuros = []
        for year in range(2025, 2032):  # 2025 hasta 2031 inclusive
            futuros.append({
                'code_num': codigo_pais,
                'year': year,
                'tourism_arrivals': tourism_arrivals,
                'tourism_expenditures': tourism_expenditures,
                'gdp': pib,
                'inflation': inflacion,
                'unemployment': desempleo
            })

        df_futuro = pd.DataFrame(futuros)

        # Hacer predicciones
        predicciones = (modelo.predict(df_futuro)) * (10**5)
        df_futuro["Predicción"] = predicciones

        # --- Crear colores personalizados según la predicción ---
        def asignar_color(valor):
            if valor <= 0:
                return "red"       # cero o menor
            elif 0 < valor <= 20:
                return "yellow"    # entre 0 y 20
            else:
                return "blue"      # mayor que 20

        df_futuro["Color"] = df_futuro["Predicción"].apply(asignar_color)

        # Crear gráfica con Plotly usando colores dinámicos
        fig = px.bar(
            df_futuro,
            x="year",
            y="Predicción",
            text="Predicción",
            labels={"year": "Año", "Predicción": "Gasto Medio (USD)"},
            title=f"Proyección del Gasto Medio por Turista en {pais} (2025-2031)",
            color="Color",
            color_discrete_map={"red": "red", "yellow": "yellow", "blue": "blue"}
        )
        fig.update_traces(
            texttemplate='%{text:,.2f}',
            textposition='outside'
        )

        fig.update_layout(showlegend=False)
        
        # Mostrar gráfica
        st.plotly_chart(fig, use_container_width=True)

# --- Mensaje final según la última predicción ---
        ultima_prediccion = df_futuro.iloc[-1]["Predicción"]
        if ultima_prediccion <= 0:
            st.error("🔴 Mala opción: la proyección indica un gasto medio nulo o negativo.")
        elif 0 < ultima_prediccion <= 20:
            st.warning("🟡 Opción de cuidado: el gasto medio proyectado es bajo, requiere precaución.")
        else:
            st.info("🔵 Buena opción: el gasto medio proyectado es alto, puede ser atractivo.")


    except Exception as e:
        st.error(f"Error al predecir: {e}")
        st.write("Datos enviados al modelo:", df_futuro)