from datetime import datetime
import os
import gymnasium as gym
import numpy as np
import time
import optuna
import matplotlib.pyplot as plt
from stable_baselines3 import DQN  # Cambiar PPO a DQN
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Asegúrate de que importas tu entorno personalizado CubeSatDetumblingEnv
from cubesat_detumbling_rl import CubeSatDetumblingEnv  # Tu entorno de CubeSat personalizado
from SatellitePersonality import SatellitePersonality  # Parámetros del satélite


# Adapter wrapper to make a Gymnasium env present the legacy Gym API
# (reset()->obs, step()->(obs, reward, done, info)). Stable-Baselines3's
# DummyVecEnv expects the old-style reset/step signatures, otherwise it may
# try to store a tuple into a numeric buffer and raise the ValueError shown.
class GymnasiumToGymWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # Gymnasium reset returns (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return obs
        return result

    def step(self, action):
        result = self.env.step(action)
        # Gymnasium step returns (obs, reward, terminated, truncated, info)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
            return obs, reward, done, info
        return result

def make_env(env_id: str, seed: int, monitor_path: str) -> gym.Env:
    """
    Crea el entorno y lo envuelve con DummyVecEnv para hacerlo compatible con Stable Baselines3
    """
    # Create the environment instance. Use render_mode=None for training to avoid GUI.
    env = CubeSatDetumblingEnv(render_mode=None)

    # Wrap to present legacy Gym API expected by some SB3 wrappers
    env = GymnasiumToGymWrapper(env)

    # Return a vectorized env for SB3
    return DummyVecEnv([lambda: env])

def save_plot_rewards_per_episode(reward_history, hyperparameters, save_dir):
    """
    Grafica las recompensas obtenidas por el agente durante cada episodio.
    
    Args:
    reward_history (list): Lista de recompensas acumuladas por episodio.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history, label="Recompensa por Episodio", color='tab:blue')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa Acumulada')
    plt.title('Recompensa por Episodio durante el Entrenamiento')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Crear un nombre de archivo basado en los hiperparámetros
    filename = f"recompensa_{hyperparameters['learning_rate']}_lr_{hyperparameters['batch_size']}_bs_{hyperparameters['gamma']}_gamma.png"
    filepath = os.path.join(save_dir, filename)
    
    # Guardar la imagen
    plt.savefig(filepath)
    plt.close()

def save_plot_success_rate(success_history, hyperparameters, save_dir):
    """
    Grafica la tasa de éxito acumulada durante el entrenamiento y guarda la imagen.
    
    Args:
    success_history (list): Lista de 1s y 0s indicando si un episodio fue exitoso (1) o no (0).
    hyperparameters (dict): Diccionario con los mejores hiperparámetros.
    save_dir (str): Directorio donde guardar la imagen.
    """
    success_rate = np.cumsum(success_history) / np.arange(1, len(success_history) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(success_rate, label="Tasa de Éxito Acumulada", color='tab:green')
    plt.xlabel('Episodio')
    plt.ylabel('Tasa de Éxito')
    plt.title('Tasa de Éxito Acumulada durante el Entrenamiento')
    plt.grid(True)
    plt.legend()

    # Crear un nombre de archivo basado en los hiperparámetros
    filename = f"tasa_exito_{hyperparameters['learning_rate']}_lr_{hyperparameters['batch_size']}_bs_{hyperparameters['gamma']}_gamma.png"
    filepath = os.path.join(save_dir, filename)
    
    # Guardar la imagen
    plt.savefig(filepath)
    plt.close()

def save_plot_average_reward_per_episode(reward_history, hyperparameters, save_dir, window_size=50):
    """
    Grafica la recompensa media por episodio con una ventana deslizante para suavizar el gráfico y guarda la imagen.
    
    Args:
    reward_history (list): Lista de recompensas acumuladas por episodio.
    hyperparameters (dict): Diccionario con los mejores hiperparámetros.
    save_dir (str): Directorio donde guardar la imagen.
    window_size (int): Tamaño de la ventana para la media móvil.
    """
    smoothed_rewards = np.convolve(reward_history, np.ones(window_size) / window_size, mode='valid')
    
    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, label=f'Recompensa Media ({window_size} episodios)', color='tab:orange')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa Media')
    plt.title(f'Recompensa Media por Episodio (Ventana {window_size})')
    plt.grid(True)
    plt.legend()

    # Crear un nombre de archivo basado en los hiperparámetros
    filename = f"recompensa_media_{hyperparameters['learning_rate']}_lr_{hyperparameters['batch_size']}_bs_{hyperparameters['gamma']}_gamma.png"
    filepath = os.path.join(save_dir, filename)
    
    # Guardar la imagen
    plt.savefig(filepath)
    plt.close()

def save_plot_success_rate_per_episode(success_history, hyperparameters, save_dir):
    """
    Grafica la tasa de éxito (1 o 0) de cada episodio y guarda la imagen.
    
    Args:
    success_history (list): Lista de 1s y 0s indicando si un episodio fue exitoso (1) o no (0).
    hyperparameters (dict): Diccionario con los mejores hiperparámetros.
    save_dir (str): Directorio donde guardar la imagen.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(success_history, label="Tasa de Éxito por Episodio", color='tab:red', linestyle='--')
    plt.xlabel('Episodio')
    plt.ylabel('Éxito (1) / No Éxito (0)')
    plt.title('Tasa de Éxito por Episodio durante el Entrenamiento')
    plt.grid(True)
    plt.legend()

    # Crear un nombre de archivo basado en los hiperparámetros
    filename = f"tasa_exito_ep_{hyperparameters['learning_rate']}_lr_{hyperparameters['batch_size']}_bs_{hyperparameters['gamma']}_gamma.png"
    filepath = os.path.join(save_dir, filename)
    
    # Guardar la imagen
    plt.savefig(filepath)
    plt.close()

def train_dqn(cfg, best_params):
    """Entrenamiento del modelo DQN"""
    os.makedirs(cfg['log_dir'], exist_ok=True)
    os.makedirs(cfg['save_dir'], exist_ok=True)

    run_id = f"{cfg['model_name']}_{cfg['env_id']}_seed{cfg['seed']}_{int(time.time())}"
    run_path = os.path.join(cfg['log_dir'], run_id)
    os.makedirs(run_path, exist_ok=True)

    monitor_path = os.path.join(run_path, "monitor.csv")

    # Semilla global (SB3 + numpy)
    set_random_seed(cfg['seed'])

    # Crear entorno y envolver con DummyVecEnv para ser compatible con Stable-Baselines3
    env = make_env(cfg['env_id'], cfg['seed'], monitor_path)

    # Definir el modelo DQN
    model = DQN("MlpPolicy", 
                env, 
                learning_rate=best_params['learning_rate'],
                batch_size=best_params['batch_size'],
                gamma=best_params['gamma'],
                verbose=1, tensorboard_log=run_path)

    # Listas para almacenar las métricas
    reward_history = []
    success_history = []

    # Entrenamiento
    for episode in range(cfg['total_timesteps']):
        obs, _ = env.reset()  # Reiniciar entorno
        done = False
        total_reward = 0

        while not done:
            # Selección de acción usando la red neuronal de DQN (usando la política greedy)
            action, _states = model.predict(obs, deterministic=True)

            # Realiza la acción y recibe la retroalimentación
            new_obs, reward, terminated, truncated, _ = env.step(action)

            # Acumulando recompensa total del episodio
            total_reward += reward
            done = terminated or truncated

            # Guardar el nuevo estado
            obs = new_obs

        # Almacenar las métricas después de cada episodio
        reward_history.append(total_reward)
        success_history.append(1 if total_reward >= 100 else 0)  # Umbral para éxito (ajustar si es necesario)

        # Actualización de la red neuronal (esto lo hace Stable-Baselines3)
        model.learn(total_timesteps=1)

    model.learn(total_timesteps=cfg['total_timesteps'])

    # Guardar el modelo entrenado
    model_path = os.path.join(cfg['save_dir'], f"{run_id}.zip")
    model.save(model_path)

    env.close()

    # Guardar los gráficos después del entrenamiento
    save_plot_rewards_per_episode(reward_history, best_params, cfg['save_dir'])
    save_plot_success_rate(success_history, best_params, cfg['save_dir'])
    save_plot_average_reward_per_episode(reward_history, best_params, cfg['save_dir'], window_size=50)
    save_plot_success_rate_per_episode(success_history, best_params, cfg['save_dir'])


    return model_path

def evaluate_model(model_path, env_id, n_eval_episodes=10):
    """Evaluar el modelo entrenado"""
    model = DQN.load(model_path)
    # Crear el entorno para evaluación (usar make_env para consistencia)
    eval_env = make_env(env_id, seed=0, monitor_path=None)

    # Evaluar el rendimiento del modelo
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)
    print(f"Evaluación del modelo: Recompensa Media: {mean_reward} ± {std_reward}")
    return mean_reward, std_reward

def objective(trial, env):
    # Sugerir hiperparámetros para DQN
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gamma = trial.suggest_float('gamma', 0.9, 0.999)

    model = DQN("MlpPolicy", 
                env, 
                learning_rate=learning_rate, 
                batch_size=batch_size, 
                gamma=gamma, 
                verbose=0)
    model.learn(total_timesteps=1000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    return mean_reward

# Función para optimizar los hiperparámetros con Optuna
def optimize_hyperparameters(env):
    # Crear un estudio de Optuna para maximizar la recompensa media
    study = optuna.create_study(direction='maximize')
    
    # Optimizar los hiperparámetros con Optuna
    study.optimize(lambda trial: objective(trial, env), n_trials=100)

    # Imprimir los mejores hiperparámetros encontrados por Optuna
    print("Best hyperparameters: ", study.best_params)

    # Retornar los mejores hiperparámetros
    return study.best_params

def main():
    # Cargar las configuraciones del satélite desde el archivo SatellitePersonality
    print(f"Nombre del satélite: {SatellitePersonality.SATELLITE_NAME}")
    print(f"Ubicación del satélite: Lat={SatellitePersonality.OBSERVER_LATITUDE}, Lon={SatellitePersonality.OBSERVER_LONGITUDE}")

    # Definir la configuración del entorno y del modelo
    cfg = {
        'env_id': 'CubeSatDetumblingEnv',  # Reemplazar con tu entorno específico de estabilización de satélite
        'total_timesteps': 100,
        'seed': 123,
        'log_dir': 'logs',
        'save_dir': 'models',
        'model_name': 'dqn_satellite',
    }

    # Crear entorno
    env = make_env(cfg['env_id'], cfg['seed'], 'monitor.csv')

    # Optimización de hiperparámetros con Optuna
    best_params = optimize_hyperparameters(env)

    # Entrenamiento del modelo
    model_path = train_dqn(cfg, best_params)
    print(f"[OK] Modelo entrenado guardado en: {model_path}")

    # Evaluar el modelo entrenado
    evaluate_model(model_path, cfg['env_id'], n_eval_episodes=10)

if __name__ == "__main__":
    main()
