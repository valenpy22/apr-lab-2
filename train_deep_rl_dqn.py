import os
import gym
import numpy as np
import time
import json
import argparse
from typing import Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

# -----------------------------
# Funciones de discretización (solo si es necesario)
# -----------------------------

def discretize_state(observation, bins):
    """Discretiza un estado continuo."""
    discretized = []
    
    for i in range(3):
        ang_vel_component = observation[4 + i]
        digit = np.digitize(ang_vel_component, bins[i])
        digit = np.clip(digit, 0, len(bins[i]) - 1)
        discretized.append(digit)
    
    n_bins = len(bins[0])
    state_index = discretized[0] * (n_bins ** 2) + discretized[1] * n_bins + discretized[2]
    
    max_state = n_bins ** 3
    if state_index >= max_state:
        state_index = max_state - 1
    
    return state_index


def create_nonuniform_bins(n_bins, max_val=1.5):
    """
    Crea bins más densos cerca de cero.
    Respeta exactamente n_bins solicitados.
    """
    if n_bins % 2 == 0:
        # Para número par: mitad negativos, mitad positivos (sin cero explícito)
        half_bins = n_bins // 2
        linear_space = np.linspace(0, 1, half_bins + 1)[1:]
        positive_bins = max_val * (linear_space ** 2)
        negative_bins = -positive_bins[::-1]
        bins = np.concatenate([negative_bins, positive_bins])
    else:
        # Para número impar: incluir cero en el centro
        half_bins = n_bins // 2
        linear_space = np.linspace(0, 1, half_bins + 1)[1:]
        positive_bins = max_val * (linear_space ** 2)
        negative_bins = -positive_bins[::-1]
        bins = np.concatenate([negative_bins, [0.0], positive_bins])
    
    return bins

# -----------------------------
# Entrenamiento del modelo
# -----------------------------

def make_env(env_id: str, seed: int, monitor_path: str) -> gym.Env:
    """
    Crea el entorno y lo envuelve con Monitor para guardar:
    - reward por episodio
    - length por episodio
    en un CSV (monitor.csv)
    """
    env = gym.make(env_id)
    env = DummyVecEnv([lambda: env])  # Wrapper para hacer compatible con Stable-Baselines3
    return env


def train_ppo(cfg):
    """Entrenamiento del modelo PPO"""
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

    # Definir el modelo PPO
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=run_path)
    model.learn(total_timesteps=cfg['total_timesteps'])

    # Guardar el modelo entrenado
    model_path = os.path.join(cfg['save_dir'], f"{run_id}.zip")
    model.save(model_path)

    env.close()
    return model_path


def evaluate_model(model_path, env_id, n_eval_episodes=10):
    """Evaluar el modelo entrenado"""
    model = PPO.load(model_path)

    # Crear el entorno para evaluación
    env = gym.make(env_id)

    # Evaluar el rendimiento del modelo
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    print(f"Evaluación del modelo: Recompensa Media: {mean_reward} ± {std_reward}")
    return mean_reward, std_reward


def main():
    # Definir la configuración del entorno y del modelo
    cfg = {
        'env_id': 'CartPole-v1',  # Reemplazar con tu entorno específico de estabilización de satélite
        'total_timesteps': 200000,
        'seed': 123,
        'log_dir': 'logs',
        'save_dir': 'models',
        'model_name': 'ppo_satellite',
    }

    # Entrenamiento del modelo
    model_path = train_ppo(cfg)
    print(f"[OK] Modelo entrenado guardado en: {model_path}")

    # Evaluar el modelo entrenado
    evaluate_model(model_path, cfg['env_id'], n_eval_episodes=10)


if __name__ == "__main__":
    main()
