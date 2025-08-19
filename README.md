## Predicción de gasto de turistas Españoles en Mundo ##

Esta aplicación predice el gasto medio por turista en distintos países del mundo, usando variables económicas (PIB, inflación, desempleo, gasto turístico, etc.) y un modelo de Machine Learning.

[![Streamlit App](https://img.shields.io/badge/🚀%20Streamlit-Live_App-FF4B4B?logo=streamlit)](https://gastomediodeturistas.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Índice 
- [Descripción](#-descripción) 
- [Demo en línea](#-demo-en-línea)
- [Instalación local](#-instalación-local)
- [Uso](#-uso)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Tecnologías utilizadas](#-tecnologías-utilizadas)
- [Contribución](#-contribución)
- [Licencia](#-licencia).

## Descripción
Este proyecto es una **aplicación interactiva en Streamlit** que predice el **gasto medio por turista** en distintos países, en base a datos económicos y turísticos como:

- PIB del país
- Número de turistas
- Gasto total turístico
- Inflación
- Desempleo

El modelo permite proyectar la evolución del gasto medio hasta el año **2031**, mostrando los resultados de forma clara y visual con **gráficas dinámicas en Plotly**.

## Demo en línea
 Prueba la aplicación directamente aquí:  
 [Streamlit App desplegada](https://gastomediodeturistas.streamlit.app/)

## Instalación local
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/marynell-rg/Prediccion_gasto_turista
   cd prediccion_gasto_turista

2. Crear entorno virtual
   python -m venv env
   source env/bin/activate   # En Linux/Mac
   env\Scripts\activate      # En Windows

3. Instalar dependencias
   pip install -r requirements.txt

4. Ejecutar la app
   streamlit run app/app.py

## Uso
1. Introduce valores de **PIB, inflación, desempleo, gasto turístico y número de turistas**.  
2. Selecciona un país y un año.  
3. Haz clic en **"Predecir gasto medio"**.  
4. Obtendrás una gráfica de barras con la proyección para los próximos 5 años.  

## Estructura del proyecto
```markdown
prediccion_gasto_turista/
│── app/
│   └── app.py
│── data/
│   ├── country_codes_num.csv
│   ├── data_tourism_crudo.csv
│   └── original_data.py
│── model/
│   └── modelo_turismo.joblib
│── notebooks/
│   ├── Estimacion_de_gasto_medio_por_turista_v1.ipynb
│   └──Estimacion_de_gasto_medio_por_turista.ipynb
│── preprocessing/
│   └── preprocessing.joblib
│── src/
│   ├── model_training.py
│   ├── preprocessing.py
│   └── transformer.py
│── assets/
│   └── turismo.jpg
│── requirements.txt
│── README.md   