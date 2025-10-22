import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# --- 1. Generación de Datos de Ejemplo ---
# Ahora simulamos dos variables: el radio del tumor y la dosis de radiación.
np.random.seed(42)
dias = np.arange(0, 100)

# El radio del tumor sigue disminuyendo
radio_real = 25 * np.exp(-dias / 50) + np.random.normal(0, 0.3, len(dias))

## MODIFICACIÓN: Creamos el historial de radiación de 46 Gy a 40 Gy
# Usamos np.linspace para crear una disminución gradual y añadimos ruido.
radiacion_real = np.linspace(46, 40, len(dias)) + np.random.normal(0, 0.15, len(dias))

# --- 2. Preprocesamiento de Datos ---

## MODIFICACIÓN: Combinamos las dos series de datos en un solo array.
# Ahora nuestro dataset tiene dos columnas (características): radio y radiación.
datos_combinados = np.stack([radio_real, radiacion_real], axis=1)

# Escalar los datos. El scaler ahora se ajustará a las dos columnas.
scaler = MinMaxScaler(feature_range=(0, 1))
datos_scaled = scaler.fit_transform(datos_combinados)

# Función para crear las secuencias. Ahora 'y' será solo la primera columna (radio).
def crear_secuencias(datos, n_pasos):
    X, y = [], []
    for i in range(len(datos) - n_pasos):
        # X ahora contiene secuencias de ambas características (radio y radiación)
        X.append(datos[i:(i + n_pasos), :]) 
        # 'y' sigue siendo solo el próximo radio a predecir (columna 0)
        y.append(datos[i + n_pasos, 0])
    return np.array(X), np.array(y)

# Definimos cuántos registros pasados usaremos para predecir el siguiente
N_PASOS = 10
X, y = crear_secuencias(datos_scaled, N_PASOS)

# La forma de X ahora es [muestras, pasos_de_tiempo, 2] porque tenemos 2 características.

# --- 3. Construcción del Modelo de Red Neuronal (LSTM) ---
model = Sequential([
    ## MODIFICACIÓN: Actualizamos el input_shape a (N_PASOS, 2)
    # El '2' le indica a la red que recibirá dos características en cada paso de tiempo.
    LSTM(50, return_sequences=True, input_shape=(N_PASOS, 2)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1) # La salida sigue siendo una neurona, ya que solo predecimos el radio.
])

# --- 4. Compilación y Entrenamiento del Modelo ---
model.compile(optimizer='adam', loss='mean_squared_error')
print("Entrenando el modelo multivariado...")
history = model.fit(X, y, epochs=100, batch_size=32, verbose=1)
print("¡Entrenamiento completo!")

# --- 5. Realización de Predicciones ---
# Predecir sobre los datos de entrenamiento
predicciones_entrenamiento_scaled = model.predict(X)

## MODIFICACIÓN: El proceso para invertir la escala es un poco más complejo.
# El scaler espera dos columnas para hacer la transformación inversa, pero el modelo solo predice una (el radio).
# Creamos un array vacío con la forma correcta y le insertamos nuestras predicciones.
predicciones_entrenamiento_unscaled = np.zeros((len(predicciones_entrenamiento_scaled), 2))
predicciones_entrenamiento_unscaled[:, 0] = predicciones_entrenamiento_scaled.flatten()
predicciones_entrenamiento = scaler.inverse_transform(predicciones_entrenamiento_unscaled)[:, 0]

# Predecir el siguiente valor
ultima_secuencia_scaled = datos_scaled[-N_PASOS:]
ultima_secuencia_scaled = ultima_secuencia_scaled.reshape(1, N_PASOS, 2) # Reshape con 2 características

prediccion_futura_scaled = model.predict(ultima_secuencia_scaled)

# Invertir la escala para la predicción futura
prediccion_futura_unscaled = np.zeros((1, 2))
prediccion_futura_unscaled[:, 0] = prediccion_futura_scaled.flatten()
prediccion_futura = scaler.inverse_transform(prediccion_futura_unscaled)[:, 0]

print(f"\nÚltimo radio conocido: {radio_real[-1]:.2f} mm con {radiacion_real[-1]:.2f} Gy")
print(f"Predicción del siguiente radio: {prediccion_futura[0]:.2f} mm")

# --- 6. Visualización de Resultados ---
fig, ax1 = plt.subplots(figsize=(14, 7))

# Gráfico del radio
color = 'tab:blue'
ax1.set_xlabel('Días')
ax1.set_ylabel('Radio del Tumor (mm)', color=color)
ax1.plot(dias, radio_real, color=color, label='Historial Real del Radio')
ax1.plot(dias[N_PASOS:], predicciones_entrenamiento, color='tab:orange', linestyle='--', label='Predicciones del Modelo')
ax1.plot(len(dias), prediccion_futura[0], 'o', color='tab:red', markersize=8, label=f'Próxima Predicción ({prediccion_futura[0]:.2f} mm)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')
ax1.grid(True)

# Gráfico de la radiación en el mismo plot con un segundo eje Y
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Dosis de Radiación (Gy)', color=color)
ax2.plot(dias, radiacion_real, color=color, linestyle=':', alpha=0.7, label='Historial de Radiación')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

fig.tight_layout()
plt.title('Predicción Multivariada del Radio del Tumor (incluyendo Radiación)')
plt.show()