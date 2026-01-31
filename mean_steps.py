import json
import numpy as np
import os

# Buscar el archivo más reciente dqn_results.json en la estructura de carpetas
def find_latest_results(base_dir="runs"):
    """Encontrar el archivo dqn_results.json más reciente."""
    latest_file = None
    latest_time = 0
    
    for root, dirs, files in os.walk(base_dir):
        if "dqn_results.json" in files:
            file_path = os.path.join(root, "dqn_results.json")
            file_time = os.path.getmtime(file_path)
            if file_time > latest_time:
                latest_time = file_time
                latest_file = file_path
    
    return latest_file

# Cargar el archivo JSON más reciente
results_file = find_latest_results()
if results_file is None:
    raise FileNotFoundError("No se encontró ningún archivo dqn_results.json")
    
print(f"Cargando resultados desde: {results_file}")
with open(results_file, 'r') as file:
    data = json.load(file)

# Extraer los pasos hasta el éxito
steps_to_success = data["final_eval"]["steps_to_success"]

# Calcular el promedio y desviación estándar de los pasos hasta el éxito
mean_steps = np.mean(steps_to_success)
std_steps = np.std(steps_to_success)

# Mostrar los resultados
print(f"Promedio de pasos hasta el éxito: {mean_steps:.2f}")
print(f"Desviación estándar de los pasos hasta el éxito: {std_steps:.2f}")
