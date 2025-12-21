from datetime import datetime
import os
os.environ['MPLBACKEND'] = 'Agg'  # Establecer el backend de Matplotlib antes de importar matplotlib
import gymnasium as gym
import numpy as np
import time
import optuna
import matplotlib
matplotlib.use('Agg', force=True)  # Usar backend no interactivo para guardar imágenes
import matplotlib.pyplot as plt
plt.ioff()  # Desactivar el modo interactivo de Matplotlib
from stable_baselines3 import DQN 
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Asegúrate de que importas tu entorno personalizado CubeSatDetumblingEnv
from cubesat_detumbling_rl import CubeSatDetumblingEnv  # Tu entorno de CubeSat personalizado
from SatellitePersonality import SatellitePersonality  # Parámetros del satélite

best_reward = -float('inf')  # Inicializa la mejor recompensa con un valor muy bajo

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

    env = Monitor(env)

    # Return a vectorized env for SB3
    return DummyVecEnv([lambda: env])


def train_dqn_return_model(env, hyperparams, total_timesteps=200000, tensorboard_log=None):
    """
    Entrena un modelo DQN con los hiperparámetros dados sobre el env y devuelve
    (model, reward_history, success_history).

    - env: vectorized env (DummyVecEnv)
    - hyperparams: dict con keys 'learning_rate','batch_size','gamma'
    - total_timesteps: número de timesteps para model.learn
    """
    # Construir el modelo
    model = DQN(
        "MlpPolicy",
        env,
        exploration_fraction=0.4, # Explora durante 40% del entrenamiento
        exploration_final_eps=0.1, # Nunca deja de explorar del todo
        learning_rate=hyperparams.get('learning_rate', 1e-3),
        batch_size=hyperparams.get('batch_size', 64),
        gamma=hyperparams.get('gamma', 0.99),
        verbose=1,
        tensorboard_log=tensorboard_log,
    )

    # Entrenar
    model.learn(total_timesteps=total_timesteps)

    # Evaluar con varios episodios para recoger reward history y success
    try:
        ep_rewards, ep_lengths = evaluate_policy(model, env, n_eval_episodes=5, return_episode_rewards=True)
        # ep_rewards is a list of episode rewards
        reward_history = list(ep_rewards)
        success_history = [1 if r >= 100 else 0 for r in reward_history]
    except Exception:
        reward_history = []
        success_history = []

    return model, reward_history, success_history

def save_plot_rewards_per_episode(reward_history, hyperparameters, save_dir):
    """
    Grafica las recompensas obtenidas por el agente durante cada episodio.
    
    Args:
    reward_history (list): Lista de recompensas acumuladas por episodio.
    """
    if not reward_history:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(reward_history, label="Recompensa por Episodio", color='tab:blue')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa Acumulada')
    plt.title('Recompensa por Episodio durante el Entrenamiento')
    plt.grid(True)
    plt.legend()

    # Crear un nombre de archivo basado en los hiperparámetros
    filename = f"recompensa_{hyperparameters['learning_rate']}_lr_{hyperparameters['batch_size']}_bs_{hyperparameters['gamma']}_gamma.png"
    filepath = os.path.join(save_dir, filename)
    
    # Guardar la imagen
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
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
    
    if not success_history:
        return
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

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
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
    if not reward_history:
        return
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

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()

def save_plot_success_rate_per_episode(success_history, hyperparameters, save_dir):
    """
    Grafica la tasa de éxito (1 o 0) de cada episodio y guarda la imagen.
    
    Args:
    success_history (list): Lista de 1s y 0s indicando si un episodio fue exitoso (1) o no (0).
    hyperparameters (dict): Diccionario con los mejores hiperparámetros.
    save_dir (str): Directorio donde guardar la imagen.
    """
    if not success_history:
        return
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

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
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

    # Entrenar el modelo usando helper (devuelve el modelo entrenado y métricas)
    model, reward_history, success_history = train_dqn_return_model(
        env, best_params, total_timesteps=cfg['total_timesteps'], tensorboard_log=run_path
    )
    
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
    try:
        eval_env.close()
    except Exception:
        pass
    print(f"Evaluación del modelo: Recompensa Media: {mean_reward} ± {std_reward}")
    return mean_reward, std_reward

def objective(trial, save_dir, cfg):
    """
    Función objetivo para optimizar los hiperparámetros con Optuna.
    En cada trial, se entrena el modelo con los hiperparámetros sugeridos y se guarda el modelo si mejora.
    
    Args:
    trial: El objeto de Optuna que sugiere los hiperparámetros.
    env: El entorno de entrenamiento.
    save_dir: El directorio donde guardar los modelos entrenados.
    cfg: La configuración del entorno y los parámetros del modelo.
    """
    global best_reward  # Acceder a la variable global best_reward

    # 1) Carpeta del trial (AQUÍ VA)
    trial_dir = os.path.join(save_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    # Sugerir hiperparámetros para DQN
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gamma = trial.suggest_float('gamma', 0.9, 0.999)

    hyperparams = {'learning_rate': learning_rate, 'batch_size': batch_size, 'gamma': gamma}

    # Create a lightweight env for this trial (use quick=True if implemented in make_env/CubeSat)
    trial_env = make_env(cfg['env_id'], seed=trial.number, monitor_path=None)

    # Train a small model for this trial
    trial_timesteps = int(cfg.get('trial_timesteps', 200))
    trial_timesteps = max(100, trial_timesteps)
    model, reward_history, success_history = train_dqn_return_model(trial_env, hyperparams, total_timesteps=trial_timesteps, tensorboard_log=None)

    # Evaluate the trained trial model
    try:
        mean_reward, std_reward = evaluate_policy(model, trial_env, n_eval_episodes=5)
    except Exception:
        mean_reward, std_reward = -float('inf'), float('inf')

    print(f"Recompensa media para lr={learning_rate}, bs={batch_size}, gamma={gamma}: {mean_reward}")

    # Update best model
    if mean_reward > best_reward:
        mp = os.path.join(trial_dir, f"best_model_{learning_rate}_{batch_size}_{gamma}.zip")
        model.save(mp)
        best_reward = mean_reward

    # Guardar gráficos después de cada trial
    save_plot_rewards_per_episode(reward_history, hyperparams, trial_dir)  # Graficar recompensas por episodio
    save_plot_success_rate(success_history, hyperparams, trial_dir)  # Graficar tasa de éxito acumulada
    save_plot_average_reward_per_episode(reward_history, hyperparams, trial_dir, window_size=50)  # Recompensa media
    save_plot_success_rate_per_episode(success_history, hyperparams, trial_dir)  # Tasa de éxito por episodio

    # cleanup trial env
    try:
        trial_env.close()
    except Exception:
        pass

    return mean_reward

# Función para optimizar los hiperparámetros con Optuna
def optimize_hyperparameters(save_dir, cfg):
    """
    Optimiza los hiperparámetros utilizando Optuna y guarda el mejor modelo entrenado.
    
    Args:
    env: El entorno de entrenamiento.
    save_dir: El directorio donde guardar los modelos.
    cfg: La configuración del entorno y los parámetros del modelo.
    """

    storage_path = os.path.join(save_dir, "optuna_dqn.db")
    storage = f"sqlite:///{storage_path}"

    study_name = "dqn_cubesat"

    # Crear un estudio de Optuna para maximizar la recompensa media
    study = optuna.create_study(direction='maximize', storage=storage, study_name=study_name, load_if_exists=True)

    n_trials = cfg.get('n_trials', 100)
    # Optimizar los hiperparámetros con Optuna
    study.optimize(lambda trial: objective(trial, save_dir, cfg), n_trials=n_trials)

    # Imprimir los mejores hiperparámetros encontrados por Optuna
    print("Best trial:", study.best_trial.number)
    print("Best value: ", study.best_value)
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
        'total_timesteps': 200000,
        'seed': 123,
        'log_dir': 'logs',
        'save_dir': 'models',
        'model_name': 'dqn_satellite',
    }

    # Optimización de hiperparámetros con Optuna (se crean envs por trial internamente)
    # reducimos trials por defecto para pruebas rápidas
    cfg.setdefault('n_trials', 10)
    cfg.setdefault('trial_timesteps', 200)
    best_params = optimize_hyperparameters(cfg['save_dir'], cfg)

    # Entrenamiento del modelo
    model_path = train_dqn(cfg, best_params)

    # Evaluar el modelo entrenado
    evaluate_model(model_path, cfg['env_id'], n_eval_episodes=10)

if __name__ == "__main__":
    main()
