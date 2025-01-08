import numpy as np
import matplotlib.pyplot as plt

# Habilitar formato LaTeX
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Datos
t_adam = np.array([1171.9, 1070.26, 1319.48, 1586.56, 1586.56])
iter_adam = np.array([12500, 8900, 13500, 14100, 14100])

t_gradient = np.array([4115.06, 3325.69, 4719.93, 4719.93, 4719.93])
iter_gradient = np.array([43100, 28000, 44900, 44900, 44900])

# Crear la figura
plt.figure(figsize=(10, 6))

# Configurar el ancho de las barras y posiciones
bar_width = 0.35  # Ancho reducido para barras más finas
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
    plt.text(bar.get_x() + bar.get_width() / 2, height / 2, f"{t_adam[i]:.1f}s", ha='center', va='center', fontsize=10, fontweight='bold', color='black')
    
for i, bar in enumerate(bars_gradient):
    height = bar.get_height()
    # Valor de iteraciones encima de la barra
    plt.text(bar.get_x() + bar.get_width() / 2, height + 500, f"{int(height)}", ha='center', va='bottom', fontsize=10)
    # Valor del tiempo dentro de la barra en negrito
    plt.text(bar.get_x() + bar.get_width() / 2, height / 2, f"{t_gradient[i]:.1f}s", ha='center', va='center', fontsize=10, fontweight='bold', color='black')

# Títulos y etiquetas
plt.title(r"\textbf{Comparación de Iteraciones y Tiempos de Optimización}", fontsize=16)
plt.xlabel(r"\textbf{Instancias (Head and Neck)}", fontsize=14)
plt.ylabel(r"\textbf{Iteraciones}", fontsize=14)
plt.xticks(x_adam + bar_width / 2, [f"Instancia {i+1}" for i in range(len(t_adam))], fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(0.18, 1), fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el gráfico
plt.tight_layout()
plt.savefig("comparacion_metodos.png", dpi=300)
plt.show()
