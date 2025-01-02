import numpy as np
import matplotlib.pyplot as plt

# Datos
t_adam = np.array([1171.9, 1070.26, 1319.48, 1586.56])
iter_adam = np.array([12500, 8900, 13500, 14100])

t_gradient = np.array([4115.06, 3325.69, 4719.93, 4719,93])
iter_gradient = np.array([43100, 28000, 44900, 44900])

# Crear la figura
plt.figure(figsize=(10, 6))

# Configurar el ancho de las barras y posiciones
bar_width = 0.2  # Ancho reducido para barras más finas
x_adam = np.arange(len(t_adam))  # Posiciones para Adam
x_gradient = x_adam + bar_width  # Desplazamiento para Gradient

# Gráfico de barras
bars_adam = plt.bar(x_adam, iter_adam, width=bar_width, color='blue', label='Adam Optimizer')
bars_gradient = plt.bar(x_gradient, iter_gradient, width=bar_width, color='red', label='Gradient Descent Optimizer')

# Mostrar valores encima de las barras y tiempos al final
for i, bar in enumerate(bars_adam):
    height = bar.get_height()
    # Valor de iteraciones encima de la barra
    plt.text(bar.get_x() + bar.get_width() / 2, height + 500, f"{int(height)}", ha='center', va='bottom', fontsize=10)
    # Valor del tiempo dentro de la barra en negrito
    plt.text(bar.get_x() + bar.get_width() / 2, height / 2, f"{t_adam[i]:.1f}s", ha='center', va='center', fontsize=10, fontweight='bold', color='white')

for i, bar in enumerate(bars_gradient):
    height = bar.get_height()
    # Valor de iteraciones encima de la barra
    plt.text(bar.get_x() + bar.get_width() / 2, height + 500, f"{int(height)}", ha='center', va='bottom', fontsize=10)
    # Valor del tiempo dentro de la barra en negrito
    plt.text(bar.get_x() + bar.get_width() / 2, height / 2, f"{t_gradient[i]:.1f}s", ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Títulos y etiquetas
plt.title("Comparación de Iteraciones y Tiempos de Optimización")
plt.xlabel("t")
plt.ylabel("Iteraciones")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el gráfico
plt.tight_layout()
plt.show()
