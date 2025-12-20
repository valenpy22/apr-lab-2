from datetime import datetime
import os
import gymnasium as gym
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

from cubesat_detumbling_rl import CubeSatDetumblingEnv


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


def save_checkpoint(q_table, ang_vel_bins, hyperparams, episode, rewards, checkpoint_dir="checkpoints"):
    """
    Guardar checkpoint del entrenamiento.
    
    Args:
        q_table: Tabla Q actual
        ang_vel_bins: Bins de discretización
        hyperparams: Diccionario de hiperparámetros
        episode: Número de episodio actual
        rewards: Lista de recompensas hasta ahora
        checkpoint_dir: Directorio para guardar checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_data = {
        'q_table': q_table,
        'ang_vel_bins': ang_vel_bins,
        'n_bins': len(ang_vel_bins[0]),
        'num_states': q_table.shape[0],
        'num_actions': q_table.shape[1],
        'hyperparameters': hyperparams,
        'episode': episode,
        'rewards': rewards,
        'timestamp': datetime.now().isoformat()
    }
    
    # Guardar checkpoint con número de episodio
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}.pkl")
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)
    
    # También guardar como "latest" para fácil recuperación
    latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pkl")
    with open(latest_path, "wb") as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"💾 Checkpoint guardado: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path):
    """
    Cargar checkpoint del entrenamiento.
    
    Returns:
        tuple: (q_table, ang_vel_bins, hyperparams, start_episode, rewards)
    """
    if not os.path.exists(checkpoint_path):
        return None
    
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"📂 Checkpoint cargado: {checkpoint_path}")
    print(f"   Episodio: {data['episode']}")
    print(f"   Timestamp: {data['timestamp']}")
    
    return (
        data['q_table'],
        data['ang_vel_bins'],
        data['hyperparameters'],
        data['episode'],
        data['rewards']
    )


def train_q_learning(episodes=10):
    """Train a Q-learning agent on the CubeSat Detumbling Environment."""
    print("=" * 50)
    print("🚀 STARTING CUBESAT DETUMBLING TRAINING WITH Q-LEARNING")
    print(f"📈 Total episodes: {episodes:,}")
    print("=" * 50)

    env = CubeSatDetumblingEnv()

    # Discretization bins for angular velocity (3 dimensions)
    # More bins = finer granularity, but larger Q-table
    n_bins = 11
    ang_vel_bins = [create_nonuniform_bins(n_bins) for _ in range(3)]
    actual_bins = len(ang_vel_bins[0]) 
    num_states = actual_bins ** 3
    print("Numero de estados:", num_states)
    num_actions = env.action_space.n

    # Initialize Q-table with zeros
    q_table = np.zeros((num_states, num_actions), dtype=np.float32)  # 4 bytes

    # Hyperparameters
    learning_rate = 0.1
    discount_factor = 0.99

    # Parámetros de temperatura para softmax
    temperature = 2.0  # Temperatura inicial (alta = más exploración)
    temperature_decay = 0.995  # Factor de decaimiento
    min_temperature = 0.01  # Temperatura mínima

    rewards = []

    for episode in range(episodes):
        print("Nuevo episodio", episode)
        obs, _ = env.reset()
        state = discretize_state(obs, ang_vel_bins)

        done = False
        total_reward = 0

        while not done:
            # Selección de acción usando softmax
            action = select_action_softmax(q_table[state, :], temperature)
            # print("Accion:", action)

            new_obs, reward, terminated, truncated, _ = env.step(action)
            # print("Velocidad angular:", new_obs[4:7])
            # print("Torque aplicado:", new_obs[10:13])

            new_state = discretize_state(new_obs, ang_vel_bins)
            # print("Nuevo estado:", new_state)
            done = terminated or truncated

            # Q-table update
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]))

            state = new_state
            total_reward += reward

        # Decaimiento de temperatura
        temperature = max(min_temperature, temperature * temperature_decay)
        rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode + 1}/{episodes} | Avg Reward (last 100): {avg_reward:.2f} | T: {temperature:.3f}")

    save_data = {
    'q_table': q_table,
    'ang_vel_bins': ang_vel_bins,
    'n_bins': len(ang_vel_bins[0]),
    'num_states': num_states,
    'num_actions': num_actions,
    'hyperparameters': {
        'learning_rate': learning_rate,
        'discount_factor': discount_factor,
        'temperature_decay': temperature_decay,
        'min_temperature': min_temperature,
        'granularity': env.sim_granularity,
        'max_steps': env.max_steps
    }
}

    # Save the Q-table
    with open("q_table8.pkl", "wb") as f:
        pickle.dump(save_data, f)
    
    print("=" * 50)
    print("💾 Q-table saved to q_table.pkl")
    print("=" * 50)
    
    env.close()
    return save_data


def train_q_learning_chunked(
    total_episodes=2000,
    chunk_size=200,
    checkpoint_dir="checkpoints",
    resume_from=None,
    save_every=100
):
    """
    Entrenar Q-learning por chunks con checkpoints automáticos.
    
    Args:
        total_episodes: Total de episodios a entrenar
        chunk_size: Episodios por chunk (para liberar recursos)
        checkpoint_dir: Directorio para checkpoints
        resume_from: Ruta a checkpoint para continuar (None = empezar nuevo)
        save_every: Guardar checkpoint cada N episodios
    """
    print("=" * 70)
    print("🚀 ENTRENAMIENTO Q-LEARNING POR CHUNKS")
    print(f"📈 Total episodios: {total_episodes:,}")
    print(f"📦 Tamaño de chunk: {chunk_size}")
    print(f"💾 Guardando cada: {save_every} episodios")
    print("=" * 70)
    
    # Intentar cargar checkpoint si existe
    if resume_from:
        checkpoint_data = load_checkpoint(resume_from)
        if checkpoint_data:
            q_table, ang_vel_bins, hyperparams, start_episode, rewards = checkpoint_data
            print(f"✅ Continuando desde episodio {start_episode}")
        else:
            print(f"⚠️ No se pudo cargar checkpoint: {resume_from}")
            print("Iniciando entrenamiento nuevo...")
            checkpoint_data = None
    else:
        checkpoint_data = None
    
    # Inicializar nuevo entrenamiento si no hay checkpoint
    if checkpoint_data is None:
        n_bins = 11
        ang_vel_bins = [create_nonuniform_bins(n_bins) for _ in range(3)]
        
        actual_n_bins = len(ang_vel_bins[0])
        num_states = actual_n_bins ** 3
        
        # Crear env temporal para obtener num_actions
        temp_env = CubeSatDetumblingEnv()
        num_actions = temp_env.action_space.n
        temp_env.close()
        
        q_table = np.zeros((num_states, num_actions))
        
        hyperparams = {
            'initial_learning_rate': 0.1,
            'min_learning_rate': 0.01,
            'learning_rate_decay': 0.9995,
            'discount_factor': 0.99,
            'initial_temperature': 1.0,
            'temperature_decay': 0.999,
            'min_temperature': 0.01,
            'max_steps': temp_env.max_steps,
            'granularity': temp_env.sim_granularity
        }
        
        start_episode = 0
        rewards = []
        
        print(f"📊 Nuevo entrenamiento:")
        print(f"   Estados: {num_states:,}")
        print(f"   Acciones: {num_actions}")
    
    # Variables de entrenamiento
    learning_rate = hyperparams['initial_learning_rate']
    temperature = hyperparams['initial_temperature']
    
    # Ajustar temperatura/LR si estamos continuando
    if start_episode > 0:
        learning_rate = max(
            hyperparams['min_learning_rate'],
            hyperparams['initial_learning_rate'] * (hyperparams['learning_rate_decay'] ** start_episode)
        )
        temperature = max(
            hyperparams['min_temperature'],
            hyperparams['initial_temperature'] * (hyperparams['temperature_decay'] ** start_episode)
        )
    
    # Entrenar por chunks
    episodes_trained = start_episode
    
    try:
        while episodes_trained < total_episodes:
            # Calcular episodios en este chunk
            episodes_in_chunk = min(chunk_size, total_episodes - episodes_trained)
            chunk_end = episodes_trained + episodes_in_chunk
            
            print(f"\n{'='*70}")
            print(f"📦 CHUNK: Episodios {episodes_trained} → {chunk_end}")
            print(f"{'='*70}")
            
            # Crear environment para este chunk
            env = CubeSatDetumblingEnv()
            
            # Entrenar chunk
            for episode in range(episodes_trained, chunk_end):
                # print("Episodio:", episode)
                # Actualizar learning rate y temperatura
                learning_rate = max(
                    hyperparams['min_learning_rate'],
                    hyperparams['initial_learning_rate'] * (hyperparams['learning_rate_decay'] ** episode)
                )
                temperature = max(
                    hyperparams['min_temperature'],
                    hyperparams['initial_temperature'] * (hyperparams['temperature_decay'] ** episode)
                )
                
                # Episodio
                obs, _ = env.reset()
                state = discretize_state(obs, ang_vel_bins)
                
                done = False
                total_reward = 0
                steps = 0
                
                while not done:
                    action = select_action_softmax(q_table[state, :], temperature)
                    new_obs, reward, terminated, truncated, _ = env.step(action)
                    new_state = discretize_state(new_obs, ang_vel_bins)
                    
                    done = terminated or truncated
                    
                    # Q-table update
                    best_next_action = np.max(q_table[new_state, :])
                    td_target = reward + hyperparams['discount_factor'] * best_next_action
                    td_error = td_target - q_table[state, action]
                    q_table[state, action] += learning_rate * td_error
                    
                    state = new_state
                    total_reward += reward
                    steps += 1
                
                rewards.append(total_reward)
                
                # Logging cada 50 episodios
                if (episode + 1) % 50 == 0:
                    recent_avg = np.mean(rewards[-50:])
                    print(f"  Ep {episode + 1:4d} | Reward: {total_reward:7.2f} | "
                          f"Avg(50): {recent_avg:7.2f} | T: {temperature:.3f} | LR: {learning_rate:.4f}")
                
                # Guardar checkpoint periódicamente
                if (episode + 1) % save_every == 0:
                    save_checkpoint(q_table, ang_vel_bins, hyperparams, episode + 1, rewards, checkpoint_dir)
            
            # Cerrar environment del chunk
            env.close()
            del env
            
            # Actualizar contador
            episodes_trained = chunk_end
            
            # Guardar checkpoint al final del chunk
            save_checkpoint(q_table, ang_vel_bins, hyperparams, episodes_trained, rewards, checkpoint_dir)
            
            # Pequeña pausa para liberar recursos
            time.sleep(1)
            
            print(f"✅ Chunk completado. Progreso: {episodes_trained}/{total_episodes} ({100*episodes_trained/total_episodes:.1f}%)")
    
    except KeyboardInterrupt:
        print("\n⚠️ Entrenamiento interrumpido por usuario")
        print("Guardando progreso actual...")
        save_checkpoint(q_table, ang_vel_bins, hyperparams, episodes_trained, rewards, checkpoint_dir)
    
    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}")
        print("Guardando progreso actual...")
        save_checkpoint(q_table, ang_vel_bins, hyperparams, episodes_trained, rewards, checkpoint_dir)
        raise
    
    # Guardar modelo final
    final_data = {
        'q_table': q_table,
        'ang_vel_bins': ang_vel_bins,
        'n_bins': len(ang_vel_bins[0]),
        'num_states': q_table.shape[0],
        'num_actions': q_table.shape[1],
        'hyperparameters': hyperparams,
        'total_episodes': episodes_trained,
        'final_rewards': rewards[-100:] if len(rewards) >= 100 else rewards
    }
    
    with open("bestbased_qtable_3.pkl", "wb") as f:
        pickle.dump(final_data, f)
    
    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print(f"📊 Total episodios: {episodes_trained:,}")
    print(f"💾 Modelo final guardado: q_table_final.pkl")
    print(f"📁 Checkpoints en: {checkpoint_dir}/")
    print("=" * 70)
    
    # Estadísticas finales
    if len(rewards) >= 100:
        print(f"\n📈 Últimas 100 recompensas:")
        print(f"   Media: {np.mean(rewards[-100]):.2f}")
        print(f"   Desv: {np.std(rewards[-100:]):.2f}")
        print(f"   Máx: {np.max(rewards[-100:]):.2f}")
        print(f"   Mín: {np.min(rewards[-100:]):.2f}")
    
    return q_table, ang_vel_bins


def list_checkpoints(checkpoint_dir="checkpoints"):
    """Listar checkpoints disponibles."""
    if not os.path.exists(checkpoint_dir):
        print(f"No existe directorio: {checkpoint_dir}")
        return []
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
    checkpoints.sort()
    
    print(f"\n📁 Checkpoints en {checkpoint_dir}:")
    print("=" * 60)
    
    for cp in checkpoints:
        cp_path = os.path.join(checkpoint_dir, cp)
        try:
            with open(cp_path, "rb") as f:
                data = pickle.load(f)
            print(f"  • {cp}")
            print(f"    Episodio: {data.get('episode', 'N/A')}")
            print(f"    Timestamp: {data.get('timestamp', 'N/A')}")
            print(f"    Tamaño: {os.path.getsize(cp_path) / 1024:.1f} KB")
            print()
        except Exception as e:
            print(f"  • {cp} (error al leer: {e})")
    
    return checkpoints


def select_action_softmax(q_values, temperature):
    """Selección de acción con softmax."""
    temperature = max(temperature, 1e-10)
    q_values_adjusted = q_values - np.max(q_values)
    exp_values = np.exp(q_values_adjusted / temperature)
    
    if np.sum(exp_values) < 1e-10:
        return np.argmax(q_values)
    
    probabilities = exp_values / np.sum(exp_values)
    probabilities /= probabilities.sum()
    action = np.random.choice(len(q_values), p=probabilities)
    
    return action


def evaluate_q_learning(q_table, episodes=1, return_stats=True):
    """Evaluate the trained Q-learning agent."""
    print("=" * 50)
    print("🧪 EVALUATING TRAINED Q-LEARNING AGENT")
    print("=" * 50)

    # Cargar modelo y metadata
    with open(q_table, "rb") as f:
        save_data = pickle.load(f)
    
    q_table = save_data['q_table']
    ang_vel_bins = save_data['ang_vel_bins']
    
    # print(f"📊 Loaded model with {save_data['num_states']:,} states")

    eval_env = CubeSatDetumblingEnv(render_mode='human', max_steps=400)
    success_count = 0
    total_rewards = []
    total_steps = []

    for episode in range(episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0
        step_count = 0

        print(f"\n🎮 Episode {episode + 1}/{episodes}")
        print("-" * 30)

        while not done and step_count < 400:
            state = discretize_state(obs, ang_vel_bins)
            action = np.argmax(q_table[state, :])
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

            if step_count % 20 == 0:
                angular_vel_norm = np.linalg.norm(obs[4:7])
                # print(f"  Step {step_count}: Angular velocity norm = {angular_vel_norm:.4f} rad/s")

        total_rewards.append(total_reward)

        if terminated:
            success_count += 1
            total_steps.append(step_count)
            print(f"  ✅ SUCCESS! Detumbling achieved in {step_count} steps")
        else:
            print(f"  ⏰ Episode ended after {step_count} steps")

        print(f"  📊 Total Reward: {total_reward:.2f}")

    eval_env.close()

    # 👉 aquí calculas los promedios
    avg_reward = float(np.mean(total_rewards))
    success_rate = success_count / episodes

    # 👉 si lo pides, devuelve los datos
    if return_stats:        
        return {
            "avg_reward": avg_reward,
            "rewards": total_rewards,
            "success_rate": success_rate,
            "avg_steps": float(np.mean(total_steps)) if total_steps else None
        }


# if __name__ == "__main__":
#     print("🛰️  CUBESAT DETUMBLING RL DEMO WITH Q-LEARNING")
#     print()
    
#     # Train the agent
#     # save_data = train_q_learning(episodes=2500)

#     # Evaluate the trained agent
#     evaluate_q_learning("q_table_final.pkl", episodes=10)
    
#     print("\n🎉 Demo completed! Check the saved model.")

def evaluate_and_plot_multiple(q_table_path, episodes_list, runs_per_episode):
    """
    Evalúa el modelo Q-learning para diferentes números de episodios de evaluación,
    ejecutando múltiples corridas para promediar la precisión, y grafica los resultados.

    Args:
        q_table_path (str): Ruta al archivo .pkl con la tabla Q entrenada.
        episodes_list (list): Lista de números de episodios de evaluación (N).
        runs_per_episode (int): Número de veces que se repite la evaluación para cada N.
    """
    print("=" * 70)
    print("📊 EVALUACIÓN MÚLTIPLE Y GRÁFICO DE PRECISIÓN (TASA DE ÉXITO)")
    print("=" * 70)

    # Cargar modelo y metadata (asumiendo que los datos de bins son necesarios)
    try:
        with open(q_table_path, "rb") as f:
            save_data = pickle.load(f)
        q_table = save_data['q_table']
        ang_vel_bins = save_data['ang_vel_bins']
        print(f"✅ Modelo cargado: {q_table_path}")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return

    results = {}
    
    # Crear environment de evaluación una vez
    eval_env = CubeSatDetumblingEnv(render_mode=None, max_steps=400)

    for num_episodes in episodes_list:
        print(f"\n--- Evaluando con N={num_episodes} episodios ---")
        success_rates = []
        
        for run in range(runs_per_episode):
            print(f"  Corrida {run + 1}/{runs_per_episode}...")
            success_count = 0
            
            for episode in range(num_episodes):
                obs, _ = eval_env.reset()
                done = False
                
                while not done:
                    state = discretize_state(obs, ang_vel_bins)
                    # La acción es la que maximiza Q (política greedy para evaluación)
                    action = np.argmax(q_table[state, :])
                    obs, _, terminated, truncated, _ = eval_env.step(action)
                    done = terminated or truncated

                if terminated:
                    success_count += 1
            
            # Tasa de éxito (precisión)
            success_rate = success_count / num_episodes
            success_rates.append(success_rate)
            print(f"    Tasa de éxito: {success_rate:.3f}")
        
        # Calcular el promedio de las tasas de éxito
        avg_success_rate = np.mean(success_rates)
        std_success_rate = np.std(success_rates)
        results[num_episodes] = {
            'mean_success_rate': avg_success_rate,
            'std_success_rate': std_success_rate
        }
        
        print(f"  ⭐ Promedio (N={num_episodes}, {runs_per_episode}x): {avg_success_rate:.3f} ± {std_success_rate:.3f}")
    
    eval_env.close()

    # Preparar datos para graficar
    episode_counts = sorted(results.keys())
    mean_rates = [results[n]['mean_success_rate'] for n in episode_counts]
    std_devs = [results[n]['std_success_rate'] for n in episode_counts]

    # Graficar
    plt.figure(figsize=(10, 6))
    plt.errorbar(episode_counts, mean_rates, yerr=std_devs, fmt='-o', capsize=5, label='Media de Tasa de Éxito Desv. Est.')
    
    plt.title('Tasa de Éxito Promedio vs. Episodios de Evaluación')
    plt.xlabel('Número de Episodios de Evaluación (N)')
    plt.ylabel('Tasa de Éxito (Precisión)')
    plt.xticks(episode_counts)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

    print("\nGráfico de precisión generado.")
    return results

def evaluate_models(q_table_paths, runs_per_episode):
    for q_table_path in q_table_paths:
        evaluate_and_plot_multiple(q_table_path, [100], runs_per_episode)

def compare_success_metrics(q_table_paths, num_episodes, num_runs=3):
    """
    Compara el rendimiento (Tasa de Éxito Media y Desv. Est.) de múltiples
    modelos Q-learning, evaluando cada uno con 'num_runs' corridas de
    'num_episodes'.

    Args:
        q_table_paths (list): Lista de rutas a los archivos .pkl de la Q-table.
        num_episodes (int): Número fijo de episodios para cada corrida.
        num_runs (int): Número de veces que se repite la evaluación para calcular la variabilidad.

    Returns:
        dict: Un diccionario con el resumen de resultados para cada modelo.
    """
    print("=" * 70)
    print(f"🏆 COMPARACIÓN DE MODELOS: TASA DE ÉXITO ({num_episodes} EPISODIOS, {num_runs} CORRIDAS)")
    print("=" * 70)
    
    results = {}
    eval_env = CubeSatDetumblingEnv(render_mode=None, max_steps=500)
    
    for path in q_table_paths:
        model_name = os.path.basename(path)
        print(f"\n--- Procesando: **{model_name}** ---")
        
        if not os.path.exists(path):
            print(f"⚠️ Archivo no encontrado: {path}. Saltando.")
            results[model_name] = {'Error': 'Archivo no encontrado'}
            continue

        try:
            # 1. Cargar el modelo
            with open(path, "rb") as f:
                data = pickle.load(f)
            q_table = data['q_table']
            ang_vel_bins = data['ang_vel_bins']
            
            success_rates = []
            
            # 2. Ejecutar múltiples corridas
            for run in range(num_runs):
                success_count = 0
                
                for episode in range(num_episodes):
                    obs, _ = eval_env.reset()
                    done = False
                    
                    while not done:
                        state = discretize_state(obs, ang_vel_bins)
                        # Política Greedy (mejor acción de la Q-table)
                        action = np.argmax(q_table[state, :])
                        obs, _, terminated, truncated, _ = eval_env.step(action)
                        done = terminated or truncated

                    if terminated:
                        success_count += 1
                
                success_rate = success_count / num_episodes
                success_rates.append(success_rate)
            
            # 3. Calcular métricas finales
            avg_success_rate = np.mean(success_rates)
            std_success_rate = np.std(success_rates)
            
            results[model_name] = {
                'Avg Success Rate': avg_success_rate,
                'Std Dev Success Rate': std_success_rate,
                'Num Episodes': num_episodes,
                'Num Runs': num_runs
            }
            
            print(f"  Resultados: Tasa de Éxito Media: {avg_success_rate:.3f} ± {std_success_rate:.3f}")

        except Exception as e:
            print(f"❌ Error durante la evaluación de {model_name}: {e}")
            results[model_name] = {'Error': str(e)}

    eval_env.close()
    
    # 4. Imprimir tabla de resumen final
    print("\n" + "=" * 70)
    print("📊 RESULTADOS FINALES DE COMPARACIÓN")
    print("-" * 70)
    
    sorted_results = sorted(results.items(), key=lambda item: item[1].get('Avg Success Rate', -1.0), reverse=True)
    
    print("{:<25} | {:<12} | {:<12}".format("Modelo", "Tasa Éxito (μ)", "Desv. Est. (σ)"))
    print("-" * 70)
    
    best_model_name = ""
    max_rate = -1.0

    for name, res in sorted_results:
        if 'Avg Success Rate' in res:
            is_best = " ⭐" if res['Avg Success Rate'] > max_rate else ""
            if res['Avg Success Rate'] > max_rate:
                max_rate = res['Avg Success Rate']
                best_model_name = name
                
            print("{:<25} | {:<12.3f} | {:<12.4f}{}".format(
                name, 
                res['Avg Success Rate'], 
                res['Std Dev Success Rate'], 
                is_best
            ))
        else:
            print("{:<25} | Error: {}".format(name, res['Error']))
    
    print("-" * 70)
    print(f"**El mejor modelo es:** **{best_model_name}**")
    print("=" * 70)

    return results

def plot_comparison_metrics(results):
    """
    Genera un gráfico de barras comparando la Tasa de Éxito Media 
    y la Desviación Estándar de los modelos evaluados.

    Args:
        results (dict): Diccionario de resultados devuelto por compare_success_metrics.
    """
    if not results:
        print("No hay resultados para graficar.")
        return

    # 1. Preparar los datos
    model_names = []
    mean_rates = []
    std_devs = []

    # Filtrar resultados válidos y extraer datos
    for name, res in results.items():
        if 'Avg Success Rate' in res:
            # Limpiar el nombre del archivo y solo tomar la parte relevante para la etiqueta
            clean_name = name.replace('bestbased_qtable_', 'M').replace('q_table5_', 'M').replace('.pkl', '')
            model_names.append(clean_name)
            mean_rates.append(res['Avg Success Rate'])
            std_devs.append(res['Std Dev Success Rate'])
    
    if not model_names:
        print("No hay datos de éxito válidos para graficar.")
        return

    # 2. Configurar y generar el gráfico
    
    # Índices para las barras
    x_pos = np.arange(len(model_names))
    
    plt.figure(figsize=(12, 7))
    
    # Crear el gráfico de barras con barras de error
    bars = plt.bar(
        x_pos, 
        mean_rates, 
        yerr=std_devs, 
        align='center', 
        alpha=0.8, 
        capsize=10,
        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    )

    # Añadir etiquetas y título
    plt.ylabel('Tasa de Éxito Media (μ)', fontsize=14)
    plt.xlabel('Modelo Q-table', fontsize=14)
    plt.title('Comparación de Modelos Q-Learning (Tasa de Éxito en 100 Episodios)', fontsize=16)
    
    # Configurar las etiquetas del eje X
    plt.xticks(x_pos, model_names, rotation=20, ha='right', fontsize=12)
    
    # Limitar el eje Y entre 0 y 1 (o un poco más)
    plt.ylim(0.0, max(max(mean_rates) * 1.1, 1.05))
    
    # 3. MODIFICACIÓN: Añadir Media y Desviación Estándar en la etiqueta
    for bar, mean_rate, std_dev in zip(bars, mean_rates, std_devs):
        height = bar.get_height()
        
        # Formato de la etiqueta: Media% ± Desv.Est.
        label_text = f'{mean_rate*100:.1f}% ± {std_dev*100:.1f}%'
        
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 label_text,
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Cuadrícula para facilitar la lectura
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout() 
    plt.show()

    print("\nGráfico de comparación de modelos generado (incluyendo sigma).")

def remove_outliers_iqr(values):
    values = np.array(values)

    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    filtered = values[(values >= lower_limit) & (values <= upper_limit)]
    return filtered, lower_limit, upper_limit



def remove_outliers_iqr(values):
    values = np.array(values)

    Q1 = np.percentile(values, 25)
    Q3 = np.percentile(values, 75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    filtered = values[(values >= lower_limit) & (values <= upper_limit)]
    return filtered, lower_limit, upper_limit



# Nota: Asegúrate de reemplazar la función original en tu script con esta versión.
if __name__ == "__main__":
    import argparse
    
    # parser = argparse.ArgumentParser(description='Entrenamiento Q-Learning por chunks')
    # parser.add_argument('--episodes', type=int, default=2000, help='Total de episodios')
    # parser.add_argument('--chunk-size', type=int, default=500, help='Episodios por chunk')
    # parser.add_argument('--save-every', type=int, default=100, help='Guardar cada N episodios')
    # parser.add_argument('--resume', type=str, default=None, help='Ruta a checkpoint para continuar')
    # parser.add_argument('--list-checkpoints', action='store_true', help='Listar checkpoints')
    
    # args = parser.parse_args()
    
    # if args.list_checkpoints:
    #     list_checkpoints()
    # else:
    #     print("🛰️ CUBESAT DETUMBLING - ENTRENAMIENTO POR CHUNKS\n")
        
    #     # Entrenar
    #     train_q_learning_chunked(
    #         total_episodes=args.episodes,
    #         chunk_size=args.chunk_size,
    #         save_every=args.save_every,
    #         resume_from=args.resume
    #     )
        
    #     print("\n🎉 Entrenamiento completado!")

    # evaluate_q_learning("bestbased_qtable_3.pkl", episodes=20)

    # q_table_file = "q_table5_best.pkl"
    # episodios_a_evaluar = [25, 50, 75, 100]
    # episodios_a_evaluar = [100]
    # corridas_por_episodio = 3

    q_table_files = ["bestbased_qtable_1.pkl", "bestbased_qtable_2.pkl", 
                    "bestbased_qtable_3.pkl", "q_table5_best.pkl"]
    
    print("🛰️ Iniciando Comparación de Modelos...")
    
    comparison_results = compare_success_metrics(
        q_table_files, 
        num_episodes=5, 
        num_runs=3
    )

    #avg = stats["avg_reward"]
    #rewards_list = stats["rewards"]
    #success = stats["success_rate"]
    #steps = stats["avg_steps"]

    #print(f"Recompensa Promedio = {avg}")
    # print(f"Recompensas = {rewards_list}")
    #print(f"Tasa de éxito = {success*100:.1f}%")
    #filtered_rewards, low, high = remove_outliers_iqr(stats["rewards"])
    #avg_filtered = np.mean(filtered_rewards)
    #print(f"Recompensa promedio sin outliers = {avg_filtered}")


    #print(f"Steps promedio = {steps}")
    # print("Recompensas sin outliers:", filtered_rewards)



    # q_table_files = ["bestbased_qtable_1.pkl", "bestbased_qtable_2.pkl", 
    #                 "bestbased_qtable_3.pkl", "q_table5_best.pkl"]
    
    # print("🛰️ Iniciando Comparación de Modelos...")
    
    # comparison_results = compare_success_metrics(
    #     q_table_files, 
    #     num_episodes=25, 
    #     num_runs=3
    # )

    # plot_comparison_metrics(comparison_results)

    # Primero, asegúrate de que el modelo exista (debe haber sido entrenado)
    """
    for q_table_file in q_table_files:
        if os.path.exists(q_table_file):
            evaluation_results = evaluate_and_plot_multiple(
                q_table_file, 
                episodios_a_evaluar, 
                corridas_por_episodio
            )
            print("\nResultados Detallados:", evaluation_results)
        else:
            print(f"Error: Archivo de tabla Q no encontrado en {q_table_file}. ¡Debes entrenar el modelo primero!")
    """
    