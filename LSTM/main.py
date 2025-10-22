import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.spatial.distance import directed_hausdorff

# ---------------- Funciones de máscara y métricas (sin cambios) ----------------
def mask_to_polygon(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [Polygon(cnt.squeeze()) for cnt in contours if len(cnt) >= 3]
    if not polys: return None
    return unary_union(polys)

def min_enclosing_circle(mask):
    points = np.column_stack(np.nonzero(mask))
    if len(points) == 0: return (None, None)
    (y, x), r = cv2.minEnclosingCircle(points.astype(np.float32))
    return (x, y), r

def max_inscribed_circle(mask):
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    _, r, _, center = cv2.minMaxLoc(dist)
    return (center[0], center[1]), r

def hausdorff_distance(mask1, mask2):
    pts1 = np.column_stack(np.nonzero(mask1))
    pts2 = np.column_stack(np.nonzero(mask2))
    if len(pts1) == 0 or len(pts2) == 0: return np.nan
    d1 = directed_hausdorff(pts1, pts2)[0]
    d2 = directed_hausdorff(pts2, pts1)[0]
    return max(d1, d2)

def compute_features(masks, doses):
    results = {}
    times = sorted(masks.keys())
    for t in times:
        mask = masks[t]
        poly = mask_to_polygon(mask)
        if poly is None: continue
        area = poly.area
        perimeter = poly.length
        center_c, R = min_enclosing_circle(mask)
        center_i, r_in = max_inscribed_circle(mask)
        results[t] = {
            "dose_applied": doses[t],
            "area": area,
            "perimeter": perimeter,
            "circum_radius": R,
            "in_radius": r_in
        }
    for i in range(len(times)-1):
        t1, t2 = times[i], times[i+1]
        if t1 in results and t2 in results:
            results[t2]["hausdorff_prev"] = hausdorff_distance(masks[t1], masks[t2])
    return results

def generate_circle_mask(shape=(100,100), center=None, radius=20):
    mask = np.zeros(shape, dtype=np.uint8)
    if center is None: center = (shape[1]//2, shape[0]//2)
    cv2.circle(mask, center, int(radius), 1, -1)
    return mask

# ---------------- Generación de datos y Modelo ----------------
def generate_training_sequences_v2(num_sequences=500, seq_len=5, initial_radius=30, min_dose=30.0, max_dose=46.0):
    X_list, Y_list = [], []
    for _ in range(num_sequences):
        current_radius = initial_radius
        radii_seq, dose_seq, factor_seq = [], [], []
        for _ in range(seq_len):
            dose = np.random.uniform(min_dose, max_dose)
            max_reduction_factor = 0.85
            min_reduction_factor = 0.98
            slope = (max_reduction_factor - min_reduction_factor) / (max_dose - min_dose)
            base_factor = min_reduction_factor + slope * (dose - min_dose)
            noise = np.random.normal(0, 0.02)
            reduction_factor = np.clip(base_factor + noise, 0.8, 1.0)
            radii_seq.append(current_radius)
            dose_seq.append(dose)
            factor_seq.append(reduction_factor)
            current_radius *= reduction_factor
        for i in range(seq_len - 1):
            X_list.append([radii_seq[i] / initial_radius, dose_seq[i] / max_dose])
            Y_list.append([factor_seq[i]])
    return torch.tensor(X_list, dtype=torch.float32), torch.tensor(Y_list, dtype=torch.float32)

class TumorPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=5):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------- Funciones de Predicción y Optimización ----------------
def predict_reduction(current_radius, dose, model, initial_radius=30.0, max_dose=46.0):
    model_input = torch.tensor([current_radius / initial_radius, dose / max_dose], dtype=torch.float32)
    with torch.no_grad():
        predicted_factor = model(model_input).item()
    return np.clip(predicted_factor, 0.0, 1.0)

def find_optimal_dose(current_radius, desired_factor, model, initial_radius=30.0, min_dose=30.0, max_dose=46.0, steps=100):
    best_dose = max_dose
    min_error = float('inf')
    for test_dose in np.linspace(min_dose, max_dose, steps):
        predicted_factor = predict_reduction(current_radius, test_dose, model, initial_radius, max_dose)
        error = abs(predicted_factor - desired_factor)
        if error < min_error:
            min_error = error
            best_dose = test_dose
    final_factor = predict_reduction(current_radius, best_dose, model, initial_radius, max_dose)
    return best_dose, final_factor

# --- Entrenamiento del modelo ---
X_data, Y_data = generate_training_sequences_v2()
model = TumorPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Entrenamiento del modelo con condición de parada ---
print("--- ENTRENANDO MODELO ---")

for epoch in range(1000): 
    optimizer.zero_grad()
    output = model(X_data)
    loss = criterion(output, Y_data)
    loss.backward()
    optimizer.step()

  
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.8f}")

# ------------ Simulación con Estrategia Híbrida ----------------
num_times = 15
initial_radius = 30.0
min_dose = 30.0
max_dose = 46.0
desired_reduction_factor = 0.90 # Objetivo para t > 1

dynamic_masks = {}
dynamic_doses = {}

# Tiempo 0: Estado inicial
dynamic_masks[0] = generate_circle_mask(radius=initial_radius)
dynamic_doses[0] = 0.0
current_radius = initial_radius

print("\n--- INICIANDO SIMULACIÓN DE TRATAMIENTO (ESTRATEGIA HÍBRIDA) ---")
for t in range(1, num_times + 1):
    dose_to_apply = 0.0
    predicted_factor = 0.0
    strategy = ""

    # Aplicamos la nueva lógica condicional
    if t == 1:
        # En el primer paso, usamos la dosis máxima
        strategy = "Máxima"
        dose_to_apply = max_dose
        predicted_factor = predict_reduction(current_radius, dose_to_apply, model)
    else:
        # Para los pasos siguientes, buscamos la dosis óptima
        strategy = "Óptima"
        dose_to_apply, predicted_factor = find_optimal_dose(current_radius, desired_reduction_factor, model)

    # Actualizamos el estado del tumor con el resultado de la estrategia elegida
    current_radius *= predicted_factor
    
    # Guardamos los resultados del tiempo t
    dynamic_masks[t] = generate_circle_mask(radius=int(np.ceil(current_radius)))
    dynamic_doses[t] = dose_to_apply
    
    print(f"Tiempo {t}: Radio Actual={current_radius:.2f} | Dosis Aplicada={dose_to_apply:.2f} Gy ({strategy}) | Reducción={predicted_factor:.3f}")

# ---------------- Calcular y mostrar métricas finales ----------------
features = compute_features(dynamic_masks, dynamic_doses)
print("\n--- RESUMEN DE MÉTRICAS FINALES ---")
for t in sorted(features.keys()):
    print(f"\nTiempo {t}:")
    for k, v in features[t].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")