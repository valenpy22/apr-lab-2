from datetime import datetime
import os
import gymnasium as gym
import numpy as np
import time
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

def train_dqn(cfg):
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
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=run_path)
    model.learn(total_timesteps=cfg['total_timesteps'])

    # Guardar el modelo entrenado
    model_path = os.path.join(cfg['save_dir'], f"{run_id}.zip")
    model.save(model_path)

    env.close()
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

    # Entrenamiento del modelo
    model_path = train_dqn(cfg)
    print(f"[OK] Modelo entrenado guardado en: {model_path}")

    # Evaluar el modelo entrenado
    evaluate_model(model_path, cfg['env_id'], n_eval_episodes=10)

if __name__ == "__main__":
    main()
