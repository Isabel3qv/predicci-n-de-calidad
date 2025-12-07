# app.py - Sistema Predictivo de Calidad de Café Arábica (Versión Final)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- 0. Definiciones y Carga de Archivos ---
FILE_PATH = "df_arabica_clean.csv"
TARGET_COLUMN = 'Overall' # COINCIDE con el entrenamiento robusto

# Cargar el Modelo y los Encoders
try:
    model = joblib.load('coffee_quality_predictor.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
except FileNotFoundError:
    st.error("🛑 ERROR: Asegúrate de que los archivos 'coffee_quality_predictor.pkl' y 'label_encoders.pkl' estén en la misma carpeta.")
    st.stop() 

# Definiciones de los rangos numéricos (deben coincidir con el entrenamiento)
NUMERIC_RANGES = {
    'Aroma': (7.0, 9.0, 8.0), 'Flavor': (7.0, 9.0, 8.0),
    'Aftertaste': (7.0, 9.0, 8.0), 'Acidity': (7.0, 9.0, 8.0),
    'Body': (7.0, 9.0, 8.0), 'Balance': (7.0, 9.0, 8.0),
    'Uniformity': (9.0, 10.0, 10.0), 'Clean Cup': (9.0, 10.0, 10.0),
    'Sweetness': (9.0, 10.0, 10.0),
    'Altitude': (500, 3000, 1500),
    'Moisture Percentage': (0.0, 0.20, 0.12)
}

# Obtener opciones de las categorías del LabelEncoder
CATEGORICAL_OPTIONS = {
    col: list(le.classes_) for col, le in label_encoders.items()
}

# --- 1. Configuración de la Aplicación Streamlit ---
st.set_page_config(page_title="Sistema Predictivo de Café", layout="wide")
st.title("☕ Sistema Predictivo de Calidad de Café Arábica")
st.markdown("---")

tab_predict, tab_viz = st.tabs(["🔮 Predicción de Puntaje", "📈 Análisis de Dataset"])

# --- 2. BARRA LATERAL PARA ENTRADA DE DATOS ---
st.sidebar.header("📝 Ingreso de Características del Lote")
input_data = {}

# 2.1. Datos de Origen y Proceso (Expanders)
with st.sidebar.expander("📍 Datos de Origen y Proceso"):
    # Recolección de Características Categóricas
    for feature, options in CATEGORICAL_OPTIONS.items():
        input_data[feature] = st.selectbox(f'{feature}', options, index=0, key=f'sb_{feature}')
    
    # Altitud y Humedad (Numéricas)
    alt_min, alt_max, alt_def = NUMERIC_RANGES['Altitude']
    input_data['Altitude'] = st.slider('Altitude (metros)', min_value=alt_min, max_value=alt_max, value=alt_def, step=10, key='sl_alt')
    moist_min, moist_max, moist_def = NUMERIC_RANGES['Moisture Percentage']
    input_data['Moisture Percentage'] = st.slider('Moisture Percentage (%)', min_value=moist_min * 100, max_value=moist_max * 100, value=moist_def * 100, step=0.1, format="%.1f%%", key='sl_moist') / 100 

# 2.2. Puntajes de Cata (Columnas)
st.sidebar.subheader("🌟 Puntajes de Cata")

col1, col2 = st.sidebar.columns(2)
# Las 9 características que el modelo usa como INPUT
cata_features_to_input = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 'Clean Cup', 'Sweetness'] 

for i, feature in enumerate(cata_features_to_input):
    min_val, max_val, default_val = NUMERIC_RANGES[feature]
    col = col1 if i < 5 else col2 

    input_data[feature] = col.slider(
        f'{feature}',
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=0.01,
        key=f'sl_{feature}'
    )

st.sidebar.markdown("---")
predict_button = st.sidebar.button('Calcular Puntaje Predicho')

# --- 3. Lógica de Predicción (Pestaña 1) ---

with tab_predict:
# app.py (Sección de Lógica de Predicción corregida)

    
    # 1. VALIDACIÓN SIMPLE DE ALTITUD
    if input_data['Altitude'] < 500:
        st.warning("⚠️ Altitud baja: Un café Arábica de especialidad se suele cultivar por encima de los 500 metros. El resultado puede no ser fiable.")

    if predict_button:
        
        # ... (Código de pre-procesamiento de entradas sigue igual) ...
        input_df = pd.DataFrame([input_data])
        
        # Pre-procesamiento de las Categóricas
        for col, le in label_encoders.items():
            try:
                value_to_encode = input_df[col].iloc[0]
                input_df[col] = le.transform([value_to_encode])[0] 
            except ValueError:
                st.warning(f"Categoría '{value_to_encode}' para {col} no reconocida. Usando valor 0.")
                input_df[col] = 0

        # Definir el ORDEN ESPERADO por el modelo
        feature_order = [
            'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 
            'Clean Cup', 'Sweetness', 
            'Altitude', 
            'Country of Origin', 
            'Variety', 
            'Processing Method', 
            'Moisture Percentage'
        ]
        # Ordenar las columnas (usando reindex para seguridad)
        try:
            input_processed = input_df.reindex(columns=feature_order, fill_value=0)
        except Exception as e:
            st.error(f"Error al reordenar columnas. Asegúrate de que el modelo fue entrenado. Error: {e}")
            st.stop() # <-- ¡CORRECCIÓN CLAVE! Usamos st.stop() en lugar de return

        # Realizar la Predicción
        prediction = model.predict(input_processed)[0]

        # ... (El resto del código para mostrar el resultado sigue igual) ...
        st.header("✨ Resultado de la Predicción")
        
        if prediction >= 85.0:
            score_style = "background-color: #4CAF50; color: white; padding: 20px; border-radius: 10px; font-size: 2.5em; text-align: center;" 
        # ... (Otros estilos) ...
        else:
            score_style = "background-color: #F44336; color: white; padding: 20px; border-radius: 10px; font-size: 2.5em; text-align: center;"

        st.markdown(f'<div style="{score_style}">Puntaje de Calidad Predicho: **{prediction:.2f}** / 100</div>', unsafe_allow_html=True)
        st.markdown("---")

        # RESUMEN DETALLADO DE LAS ENTRADAS
        st.subheader("📋 Resumen de las Características Ingresadas")
        
        input_summary = pd.DataFrame.from_dict(input_data, orient='index', columns=['Valor Ingresado'])
        if 'Overall' in input_summary.index:
            input_summary = input_summary.drop(index='Overall') 
            
        input_summary.loc['Moisture Percentage', 'Valor Ingresado'] = f"{input_data['Moisture Percentage'] * 100:.1f}%"
        
        st.dataframe(input_summary)
        
    else:
        st.info("Ajusta los parámetros en la barra lateral izquierda y haz clic en 'Calcular Puntaje Predicho'.")
# --- 4. Análisis de Dataset (Pestaña 2) ---

with tab_viz:
    st.header("Análisis del Dataset de Calidad de Café")
    # ... (El código de visualización de Plotly sigue aquí) ...
    try:
        df_viz = pd.read_csv(FILE_PATH)
        df_viz['Altitude'] = pd.to_numeric(df_viz['Altitude'], errors='coerce')
        df_viz.dropna(subset=[TARGET_COLUMN, 'Country of Origin', 'Altitude'], inplace=True)

        st.subheader("Distribución de Puntaje por País")
        fig_country = px.box(df_viz, x='Country of Origin', y=TARGET_COLUMN, title='Distribución de Puntajes por País de Origen', labels={TARGET_COLUMN: "Puntaje Total", "Country of Origin": "País"}, color='Country of Origin')
        st.plotly_chart(fig_country, use_container_width=True)

        st.subheader("Relación entre Altitud y Puntaje")
        fig_altitude = px.scatter(df_viz, x='Altitude', y=TARGET_COLUMN, hover_data=['Country of Origin', 'Variety'], title='Puntaje Total vs. Altitud', labels={'Altitude': 'Altitud (metros)', TARGET_COLUMN: 'Puntaje Total'})
        st.plotly_chart(fig_altitude, use_container_width=True)

    except Exception as e:
        st.warning(f"No se pudo cargar o visualizar el dataset para el análisis. Asegúrate de que '{FILE_PATH}' existe. Error: {e}")