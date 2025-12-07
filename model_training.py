# model_training.py - VERSIÓN FINAL Y ROBUSTA (Ajustada para predecir 'Overall')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib 
import re 

# --- 1. Definiciones y Carga ---
FILE_PATH = "df_arabica_clean.csv"
TARGET_COLUMN = 'Overall' 

# 'Overall' se elimina de FEATURES porque ahora es la TARGET
FEATURES = [
    'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 
    'Clean Cup', 'Sweetness', 
    'Altitude', 
    'Country of Origin', 
    'Variety', 
    'Processing Method', 
    'Moisture Percentage'
]
categorical_cols = ['Country of Origin', 'Variety', 'Processing Method']
numeric_cols = [f for f in FEATURES if f not in categorical_cols]


try:
    df = pd.read_csv(FILE_PATH)
    print(f"✅ Dataset '{FILE_PATH}' cargado exitosamente. Filas iniciales: {len(df)}")
except FileNotFoundError:
    print(f"🛑 ERROR: Archivo '{FILE_PATH}' no encontrado.")
    exit()

# Validación de columnas y creación de df_model
required_cols = FEATURES + [TARGET_COLUMN]
for col in required_cols:
    if col not in df.columns:
        print(f"🛑 ERROR: La columna requerida '{col}' no existe en el dataset. Revisa la lista FEATURES.")
        exit()

df_model = df[required_cols].copy()

# Forzar la columna objetivo a ser float y revisar el estado
df_model[TARGET_COLUMN] = pd.to_numeric(df_model[TARGET_COLUMN], errors='coerce')
print(f"\n📢 Revisando la columna objetivo '{TARGET_COLUMN}':")
print(f"   Valores NO nulos encontrados: {df_model[TARGET_COLUMN].notnull().sum()} de {len(df_model)}")


# --- 2. Limpieza de Datos y Conversión (Imputación Total) ---

print("\n⏳ Corrigiendo Altitud e imputando todas las columnas...")

# FUNCIÓN para limpiar y extraer Altitud (Misma función robusta)
def clean_altitude(alt):
    if pd.isna(alt): return np.nan
    try: return float(alt)
    except:
        alt = str(alt).lower().replace(',', '').strip()
        if '-' in alt:
            parts = alt.split('-');
            if len(parts) == 2:
                try: return (float(parts[0]) + float(parts[1])) / 2
                except: pass
        match = re.search(r'(\d+)', alt);
        if match: return float(match.group(1));
        return np.nan 

# Aplicar limpieza de Altitud
df_model.loc[:, 'Altitude'] = df_model['Altitude'].apply(clean_altitude)


# ELIMINACIÓN: Quitamos las filas donde no haya puntaje (Target)
df_model.dropna(subset=[TARGET_COLUMN], inplace=True) 
print(f"   Filas restantes después de eliminar nulos del puntaje: {len(df_model)}")


# IMPUTACIÓN DE VALORES (Rellenar nulos en Features)

# Imputar Categóricas: Rellenar nulos con un marcador de posición 'N/A_Missing'
for col in categorical_cols:
    df_model.loc[:, col] = df_model[col].fillna('N/A_Missing').astype(str)

# Imputar Numéricas: Rellenar nulos con la media
for col in numeric_cols:
    if df_model[col].isnull().any():
        try:
            mean_value = df_model[col].mean()
            df_model.loc[:, col] = df_model[col].fillna(mean_value)
        except TypeError:
            df_model.loc[:, col] = df_model[col].fillna(0) 


print(f"✅ Filas finales después de la limpieza e imputación: {len(df_model)}")

# --- 3. Pre-procesamiento de Características Categóricas ---

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model.loc[:, col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

# Guardamos los encoders
joblib.dump(label_encoders, 'label_encoders.pkl')
print("✅ Encoders guardados en 'label_encoders.pkl'.")

# --- 4. Entrenamiento del Modelo (Scikit-learn) ---

X = df_model.drop(columns=[TARGET_COLUMN])
y = df_model[TARGET_COLUMN]

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Datos divididos. Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")
print("⏳ Entrenando RandomForestRegressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("✅ Entrenamiento completado.")

score = model.score(X_test, y_test)
print(f"📊 R2 Score del modelo en datos de prueba: {score:.4f}")

# --- 5. Guardar el Modelo Entrenado ---
MODEL_FILENAME = 'coffee_quality_predictor.pkl'
joblib.dump(model, MODEL_FILENAME)
print(f"✅ Modelo guardado en '{MODEL_FILENAME}'.")