## Predicción de gasto de turistas Españoles en Mundo ##

Esta aplicación predice el gasto medio por turista en distintos países del mundo, usando variables económicas (PIB, inflación, desempleo, etc.) y un modelo de Machine Learning.

[![Streamlit App](https://img.shields.io/badge/🚀%20Streamlit-Live_App-FF4B4B?logo=streamlit)](https://gastomediodeturistas.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Índice 
- [Descripción](#descripcion) 
- [Demo en línea](#demo-en-linea)
- [Instalación local](#instalacion-local)
- [Uso](#uso)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Tecnologías utilizadas](#tecnologias-utilizadas)
- [Contribución](#contribucion)
- [Licencia](#licencia)
- [Autoría](#autoria)

## Descripción
Este proyecto es una **aplicación interactiva en Streamlit** que predice el **gasto medio por turista** en distintos países, en base a datos económicos y turísticos como:

- Gasto total turístico
- Número de turistas
- PIB del país
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
   ```bash
   python -m venv env
   source env/bin/activate        # En Linux/Mac
   .env\Scripts\Activate.ps1      # En Windows

3. Instalar dependencias
   ```bash
   pip install -r requirements.txt

4. Ejecutar la app
   ```bash
   streamlit run app/app.py

## Uso
1. Introduce valores de **gasto turístico, número de turistas, PIB, inflación y desempleo**.  
2. Selecciona un país y un año.  
3. Haz clic en **"Predecir gasto medio"**.  
4. Obtendrás una gráfica de barras con la proyección para los próximos 5 años.  

## Estructura del proyecto
   ```bash
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
   ```

## Tecnologías utilizadas
   ```bash
   - joblib==1.5.1
   - numpy==2.3.0
   - pandas==2.3.0
   - python-dateutil==2.9.0.post0
   - pytz==2025.2
   - scikit-learn==1.7.0
   - scipy==1.15.3
   - six==1.17.0
   - threadpoolctl==3.6.0
   - tzdata==2025.2
   - streamlit>=1.26.0
   - rich>=10.14.0
   - kagglehub
   - plotly
   ```

## Contribución
¡Las contribuciones son bienvenidas!

Si deseas colaborar:
1. Haz un fork del proyecto.
2. Crea una rama con tu mejora: git checkout -b feature/nueva-funcionalidad.
3. Haz commit de los cambios: git commit -m 'Añadir nueva funcionalidad'.
4. Haz push a la rama: git push origin feature/nueva-funcionalidad.
5. Abre un Pull Request.

## Licencia
```bash
Este proyecto está bajo la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
```

## Autoría
Proyecto desarrollado por **Marynell Romero** como parte de un ejercicio de predicción y visualización de la relación entre turismo y economía.