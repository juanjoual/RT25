import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import joblib

# --- FASE 1: Preparación de Datos ---
def load_and_prepare_data(filepath='historical_plans.csv'):
    """
    Carga los datos y los prepara para el entrenamiento.
    En un escenario real, cargarías tu CSV aquí.
    Ahora, generamos datos ficticios para que el código se pueda ejecutar.
    """
    print("Generando datos ficticios para demostración...")
    
    # LISTA DEFINITIVA DE ROIs (incluyendo Cócleas)
    rois = [
        "Patient", "Spinal Cord", "Parotid (R)", "Parotid (L)", "SMG (R)", "SMG (L)",
        "MCS", "MCM", "MCI", "MCP", "Oesophagus", "Brainstem", "Oral Cavity",
        "Larynx", "PTV 0-46Gy", "Cochlea (L)", "Cochlea (R)", "PTV Shell 15mm", 
        "PTV Shell 30mm", "PTV Shell 40mm", "PTV Shell 5mm", "PTV Shell 0mm", 
        "Ext. Ring 20mm"
    ]
    num_patients = 150 # Aumentado para más estabilidad
    
    data = []
    for i in range(num_patients):
        ptv_vol = np.random.uniform(100, 250)
        for roi_name in rois:
            # --- Características (Features - X): LA ANATOMÍA ---
            features = {
                "patient_id": i,
                "roi_index": rois.index(roi_name),
                "ptv_volume": ptv_vol,
                "roi_volume": np.random.uniform(2, 50) if "PTV" not in roi_name else ptv_vol,
                "dist_ptv_roi_mean": np.random.uniform(0.1, 20) if "PTV" not in roi_name else 0,
                "overlap_ptv_roi_vol": np.random.uniform(0, 8) if "PTV" not in roi_name else ptv_vol,
            }
            
            # --- Etiquetas (Labels - Y): LOS PARÁMETROS DE OPTIMIZACIÓN ---
            is_target = 1 if "PTV 0-46Gy" in roi_name else 0
            max_edose = -1.0
            if "Spinal Cord" in roi_name or "Brainstem" in roi_name:
                max_edose = 38.0 + np.random.randn()
            elif "Shell" not in roi_name and "Ring" not in roi_name and not is_target:
                 max_edose = 48.3 + np.random.randn()

            min_geud = -10.0 if is_target else 10.0 + np.random.randn()
            max_geud = 10.0 if is_target else 5.0 + np.random.randn()
            
            labels = {
                "param_is_target": float(is_target),
                "param_min_dose": 46.0 if is_target else -1.0,
                "param_max_eDose": max_edose,
                "param_mean_eDose": max_edose, # Simplificación para el ejemplo
                "param_min_gEUD": min_geud,
                "param_max_gEUD": max_geud
            }
            data.append({**features, **labels})
            
    df = pd.DataFrame(data)
    feature_cols = [col for col in df.columns if not col.startswith('param_') and col != 'patient_id']
    label_cols = [col for col in df.columns if col.startswith('param_')]
    
    return df[feature_cols], df[label_cols], feature_cols, label_cols

# --- FASE 2: Entrenamiento del Modelo ---
def train_model(X, y):
    print("\n--- Fase de Entrenamiento Iniciada ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    base_estimator = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model = MultiOutputRegressor(estimator=base_estimator)
    print("Entrenando el modelo... (puede tardar un momento)")
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Puntuación R^2 del modelo en el conjunto de prueba: {score:.4f}")
    return model

# --- FASE 3: Predicción para un Nuevo Paciente ---
def get_manual_patient_features():
    """
    Define manualmente las características GEOMÉTRICAS (los "ingredientes") 
    para el paciente que quieres predecir.
    
    ¡¡¡ACCIÓN REQUERIDA!!!:
    Extrae estos valores de tu Sistema de Planificación (TPS) para el paciente 
    "Head-and-Neck 01" y REEMPLAZA los valores de ejemplo de abajo.
    """
    print("\n--- Cargando Anatomía Manual del Paciente ---")
    rois = [
        "Patient", "Spinal Cord", "Parotid (R)", "Parotid (L)", "SMG (R)", "SMG (L)",
        "MCS", "MCM", "MCI", "MCP", "Oesophagus", "Brainstem", "Oral Cavity",
        "Larynx", "PTV 0-46Gy", "Cochlea (L)", "Cochlea (R)", "PTV Shell 15mm", 
        "PTV Shell 30mm", "PTV Shell 40mm", "PTV Shell 5mm", "PTV Shell 0mm", 
        "Ext. Ring 20mm"
    ]
    # Característica clave: Volumen del PTV principal en cc
    ptv_vol_new = 210.0

    # Diccionario con los datos ANATÓMICOS del paciente. ¡EDITA ESTO!
    patient_data_map = {
        # ROI_NAME:           {"vol": roi_volume_cc, "dist": mean_dist_ptv_mm, "overlap": overlap_vol_cc}
        "Patient":            {"vol": 5000.0, "dist": 0.0, "overlap": ptv_vol_new},
        "Spinal Cord":        {"vol": 25.0,   "dist": 5.5, "overlap": 0.0},
        "Parotid (R)":        {"vol": 30.0,   "dist": 1.2, "overlap": 0.5},
        "Parotid (L)":        {"vol": 32.0,   "dist": 1.1, "overlap": 0.6},
        "SMG (R)":            {"vol": 12.0,   "dist": 0.2, "overlap": 8.0},
        "SMG (L)":            {"vol": 13.0,   "dist": 0.2, "overlap": 9.0},
        "MCS":                {"vol": 8.0,    "dist": 0.1, "overlap": 7.5},
        "MCM":                {"vol": 9.0,    "dist": 0.1, "overlap": 8.5},
        "MCI":                {"vol": 10.0,   "dist": 0.1, "overlap": 9.5},
        "MCP":                {"vol": 11.0,   "dist": 0.5, "overlap": 5.0},
        "Oesophagus":         {"vol": 40.0,   "dist": 1.5, "overlap": 0.0},
        "Brainstem":          {"vol": 20.0,   "dist": 8.0, "overlap": 0.0},
        "Oral Cavity":        {"vol": 150.0,  "dist": 0.8, "overlap": 15.0},
        "Larynx":             {"vol": 60.0,   "dist": 0.2, "overlap": 10.0},
        "PTV 0-46Gy":         {"vol": ptv_vol_new, "dist": 0.0, "overlap": ptv_vol_new},
        "Cochlea (L)":        {"vol": 1.5,    "dist": 15.0, "overlap": 0.0}, # Añadido
        "Cochlea (R)":        {"vol": 1.6,    "dist": 16.0, "overlap": 0.0}, # Añadido
        "PTV Shell 15mm":     {"vol": 350.0,  "dist": 0.0, "overlap": 0.0},
        "PTV Shell 30mm":     {"vol": 500.0,  "dist": 0.0, "overlap": 0.0},
        "PTV Shell 40mm":     {"vol": 650.0,  "dist": 0.0, "overlap": 0.0},
        "PTV Shell 5mm":      {"vol": 250.0,  "dist": 0.0, "overlap": 0.0},
        "PTV Shell 0mm":      {"vol": 220.0,  "dist": 0.0, "overlap": 0.0},
        "Ext. Ring 20mm":     {"vol": 800.0,  "dist": 0.0, "overlap": 0.0},
    }

    new_patient_features = []
    for i, roi_name in enumerate(rois):
        roi_data = patient_data_map.get(roi_name, {"vol": 0.0, "dist": 0.0, "overlap": 0.0})
        features = {
            "roi_index": i,
            "ptv_volume": ptv_vol_new,
            "roi_volume": roi_data["vol"],
            "dist_ptv_roi_mean": roi_data["dist"],
            "overlap_ptv_roi_vol": roi_data["overlap"],
        }
        new_patient_features.append(features)
    return new_patient_features, rois

def predict_for_new_patient(model, new_patient_features, rois, feature_cols, label_cols):
    """
    Realiza la predicción (la "receta") para la anatomía dada.
    """
    print("\n--- Fase de Predicción Iniciada ---")
    new_patient_df = pd.DataFrame(new_patient_features, columns=feature_cols)
    predicted_params = model.predict(new_patient_df)
    predicted_df = pd.DataFrame(predicted_params, columns=label_cols)
    
    print("\n--- Parámetros de Optimización gEUD Predichos ---")
    for i, roi_name in enumerate(rois):
        params = predicted_df.iloc[i]
        is_target_pred = bool(round(params['param_is_target']))
        min_dose_pred = params['param_min_dose']
        max_edose_pred = params['param_max_eDose']
        mean_edose_pred = params['param_mean_eDose']
        min_geud_pred = params['param_min_gEUD']
        max_geud_pred = params['param_max_gEUD']
        
        print(f'plan.regions[{i:2d}].set_targets({str(is_target_pred).lower():5s}, {min_dose_pred:6.2f},    -1,    -1, {max_edose_pred:5.2f}, {mean_edose_pred:5.2f}, {min_geud_pred:4.1f}, {max_geud_pred:4.1f}); // {roi_name}')

if __name__ == '__main__':
    # 1. Preparar datos (en este caso, generar datos ficticios de entrenamiento)
    X, y, feature_cols, label_cols = load_and_prepare_data()
    
    # 2. Entrenar el modelo y guardarlo
    trained_model = train_model(X, y)
    joblib.dump(trained_model, 'gEUD_parameter_predictor.pkl')
    print("\nModelo entrenado y guardado como 'gEUD_parameter_predictor.pkl'")
    
    # 3. Cargar el modelo guardado (simulando un nuevo uso)
    loaded_model = joblib.load('gEUD_parameter_predictor.pkl')

    # 4. Cargar la anatomía del paciente específico que queremos predecir
    manual_features, rois_for_prediction = get_manual_patient_features()

    # 5. Usar el modelo para predecir los parámetros para ese paciente
    predict_for_new_patient(
        loaded_model, 
        manual_features, 
        rois_for_prediction, 
        feature_cols, 
        label_cols
    )

