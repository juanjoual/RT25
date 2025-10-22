import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time

# --- PARÁMETROS CLAVE (AJUSTADOS PARA UN MEJOR APRENDIZAJE) ---
HISTORY_LENGTH = 10
N_FUTURE_STEPS = 3
INITIAL_RADIUS = 40.0
MAX_DOSE = 46.0

# --- PASO 1: Generación de Datos (Con la relación Dosis-Efecto bien definida) ---
def generate_historical_sequences(num_sequences=1000, seq_len=20, history_len=HISTORY_LENGTH):
    X_list, Y_list = [], []
    for _ in range(num_sequences):
        radii = [INITIAL_RADIUS]
        doses = [0.0]
        factors = []
        current_radius = INITIAL_RADIUS
        for _ in range(seq_len):
            dose = np.random.uniform(30.0, MAX_DOSE)
            
            # Relación fuerte y limpia: Dosis altas = Reducción alta (factor bajo)
            max_reduction, min_reduction = 0.80, 1.0
            slope = (max_reduction - min_reduction) / (MAX_DOSE - 30.0)
            base_factor = min_reduction + slope * (dose - 30.0)
            noise = np.random.normal(0, 0.005)
            reduction_factor = np.clip(base_factor + noise, 0.78, 1.0)

            current_radius *= reduction_factor
            radii.append(current_radius)
            doses.append(dose)
            factors.append(reduction_factor)
            
        for i in range(len(radii) - history_len - 1):
            history_slice = []
            for j in range(history_len):
                norm_radius = radii[i+j] / INITIAL_RADIUS
                norm_dose = doses[i+j] / MAX_DOSE
                history_slice.append([norm_radius, norm_dose])
            X_list.append(history_slice)
            Y_list.append([factors[i + history_len - 1]])
            
    return torch.tensor(X_list, dtype=torch.float32), torch.tensor(Y_list, dtype=torch.float32)

# --- PASO 2: Modelo LSTM (CON MAYOR CAPACIDAD) ---
class HistoricalTumorLSTM(nn.Module):
    # Aumentamos el tamaño de la capa oculta para darle más capacidad de aprendizaje.
    def __init__(self, input_size=2, hidden_size=100, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1) # Añadimos dropout
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# --- PASO 3: Funciones de Predicción y Optimización (Sin cambios) ---
def predict_with_history(history, dose_to_test, model):
    model.eval()
    input_sequence_data = list(history)[:-1]
    current_radius = history[-1][0]
    hypothetical_action = (current_radius, dose_to_test)
    input_sequence_data.append(hypothetical_action)
    
    model_input_list = []
    for r, d in input_sequence_data:
        model_input_list.append([r / INITIAL_RADIUS, d / MAX_DOSE])
    
    model_input = torch.tensor([model_input_list], dtype=torch.float32)
    
    with torch.no_grad():
        predicted_factor = model(model_input).item()
        
    return np.clip(predicted_factor, 0.8, 1.0)

def find_optimal_dose_with_history(history, desired_factor, model, steps=100):
    best_dose = MAX_DOSE
    min_error = float('inf')
    
    for test_dose in np.linspace(30.0, MAX_DOSE, steps):
        predicted_factor = predict_with_history(list(history), test_dose, model)
        error = abs(predicted_factor - desired_factor)
        
        if error < min_error:
            min_error = error
            best_dose = test_dose
            
    final_factor = predict_with_history(list(history), best_dose, model)
    
    return best_dose, final_factor

# --- PASO 4: Entrenamiento del Modelo (CON HIPERPARÁMETROS AJUSTADOS) ---
print(f"--- ENTRENANDO MODELO CON HISTORY_LENGTH = {HISTORY_LENGTH} ---")
X_data, Y_data = generate_historical_sequences()
model = HistoricalTumorLSTM()
criterion = nn.MSELoss()
# Optimizador con una tasa de aprendizaje más baja y paciente.
optimizer = optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()
model.train()
# Aumentamos las épocas de entrenamiento.
for epoch in range(2000):
    optimizer.zero_grad()
    output = model(X_data)
    loss = criterion(output, Y_data)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.8f}")
end_time = time.time()
print(f"\nEntrenamiento completado en {end_time - start_time:.2f} segundos.")

# --- PASO 5: Guardar el Modelo Entrenado ---
torch.save(model.state_dict(), 'modelo_lstm_cpu_final.pth')
print("Modelo entrenado y guardado en 'modelo_lstm_cpu_final.pth'")

# --- PASO 6: Proyección Futura ---
historial_sintetico = [
    (40.0, 0.0),    # t-10
    (38.0, 40.0),   # t-9
    (36.5, 42.0),   # t-8
    (34.0, 45.0),   # t-7
    (32.0, 46.0),   # t-6
    (30.0, 0.0),    # t-5
    (28.0, 46.0),   # t-4
    (26.5, 46.0),   # t-3
    (25.0, 44.0),   # t-2
    (23.0, 44.0),   # t-1
    (21.0, 42.0)    # t (actual)
]

print("\n" + "="*60)
print("INICIANDO PROYECCIÓN FUTURA BASADA EN HISTORIAL SINTÉTICO")
print("="*60)

projection_history = deque(historial_sintetico[-HISTORY_LENGTH:], maxlen=HISTORY_LENGTH)
current_radius = historial_sintetico[-1][0]
desired_reduction_factor = 0.88

print(f"--- Historial Conocido (los últimos {HISTORY_LENGTH} puntos) ---")
for i in range(len(projection_history)):
    r, d = projection_history[i]
    print(f"Tiempo t-{HISTORY_LENGTH-1-i}: Radio={r:.2f}, Dosis={d:.2f} Gy")

print("\n--- Proyección Futura (t+1 a t+3) ---")
print(f"Objetivo de Reducción: {desired_reduction_factor}")
for i in range(1, N_FUTURE_STEPS + 1):
    dose_ideal, predicted_factor = find_optimal_dose_with_history(
        projection_history, desired_reduction_factor, model
    )
    
    next_radius = current_radius * predicted_factor
    print(f"Tiempo t+{i}: Radio Futuro={next_radius:.2f} | Dosis Ideal={dose_ideal:.2f} Gy | Reducción={predicted_factor:.3f}")
    
    current_radius = next_radius
    projection_history.append((current_radius, dose_ideal))