# Librerías a utilizar
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time


import matplotlib
matplotlib.use("TkAgg")  # Backend interactivo
import matplotlib.pyplot as plt
import matplotlib
from zmq.backend import second

plt.close('all')

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Importar componentes existentes del simulador de HoneySat
from Simulations.RotationSimulation import RotationSimulation
from Simulations.OrbitalSimulation import OrbitalSimulation
from Simulations.MagneticSimulation import MagneticSimulation
from SatellitePersonality import SatellitePersonality


class CubeSatDetumblingEnv(gym.Env):
    """
    ENTORNO DE GYMNASIUM PARA PROBLEMA DE DETUMBLING USANDO SIMULADOR DE HONEYSAT.

    Este entorno se integra con las clases existentes de RotationSimulation, OrbitalSimulation
    y MagneticSimulation para proporcionar una simulación realista de la dinámica de satélites
    para el aprendizaje por refuerzo.
    """

    metadata = {'render_modes': ['human', 'none']}

    def __init__(self, render_mode=None, max_steps=400, start_time=datetime.now(), time_step=0.1, granularity=40, debug=False, num_bins=4, plot_hist=False):
        """
        Inicializar el entorno de CubeSat para el problema de detumbling.

        Args:
            render_mode (str): Modo de renderizado ('human' o None)
            max_steps (int): Pasos máximos por episodio
            start_time (datetime): Tiempo inicial de la simulacion
            time_step (float): Paso de tiempo de simulación en segundos
            granularity (int): Granularida de la simulacion, divide a time_step
            debug (bool): Activar historico de observaciones y graficar
        """
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.time_step = time_step
        self.current_time = start_time
        self.sim_granularity = granularity
        self._plot_hist = plot_hist

        # inicializar componentes del simulador
        self.rotation_sim = None
        self.orbital_sim = None
        self.magnetic_sim = None

        self.num_bins = num_bins

        # Discretize the action space for Q-learning
        # Actions: Positive/Negative torque on each axis (X, Y, Z) + No torque
        self.max_torque = SatellitePersonality.MAX_TORQUE_REACTION_WHEEL
        self.action_map = self.create_action_map_xyz()
        self.action_space = spaces.Discrete(len(self.action_map))

        # definir espacio de observaciones
        # box: quaternion (4) + velocidad angular (3) + campo magnetico (3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )

        # tracking dentro de un episodio
        self.current_step = 0
        self.episode_reward = 0.0

        # como es casi imposible tener velocidad angular cero, se establece un umbral
        # ajustable dependiendo de la misión y el contexto
        self.success_threshold = 0.01  # rad/s

        # Para efectos de debug, guardar un historial de observaciones y graficarlas
        self._debug = debug  # Debug activado
        self._observation_hist = []  # Historico de observaciones
        self._time_hist = []  # Historic time
        if self._debug:
            # import matplotlib.pyplot as plt
            pass
            # self.__figure, axes = plt.subplots(2, 1)
            # axes[0].grid(True)
            # axes[1].grid(True)
            # plt.ion()
            # plt.show(block=False)
    
    def create_action_map_xyz(self):
        """
        Crear el action_map para los tres ejes X, Y y Z usando discretización logarítmica.
        Más bins cerca de cero para control fino.
        """
        action_map = {}

        min_torque = self.max_torque
        max_torque = self.max_torque

        # Crear bins logarítmicos para un solo eje
        positive_bins = np.linspace(min_torque, max_torque, self.num_bins // 4)
        negative_bins = -positive_bins[::-1]  # los negativos

        positive_bins_z = np.linspace(max_torque / 8, max_torque, self.num_bins // 2)
        negative_bins_z = -positive_bins_z[::-1]  # los negativos

        # Incluimos el cero
        axis_bins = np.concatenate([negative_bins, positive_bins])
        axis_bins_z = np.concatenate([negative_bins_z, positive_bins_z])

        # print(axis_bins)

        # Crear todas las combinaciones posibles de X, Y y Z
        index = 0

        for tx in axis_bins:
            action_map[index] = np.array([tx, 0, 0])
            print(action_map[index])
            index += 1
        
        for ty in axis_bins:
            action_map[index] = np.array([0, ty, 0])
            print(action_map[index])
            index += 1

        for tz in axis_bins_z:
            action_map[index] = np.array([0, 0, tz])
            print(action_map[index])
            index += 1

        # print(action_map)
        return action_map


    def _create_simulators(self):
        """Crear instancias de simuladores.
        - RotationSimulation
        - OrbitalSimulation
        - MagneticSimulation
        """
        if self.rotation_sim is not None:
            try:
                self.rotation_sim.stop()
            except Exception:
                pass
        if self.orbital_sim is not None:
            try:
                self.orbital_sim.stop()
            except Exception:
                pass
        if self.magnetic_sim is not None:
            try:
                self.magnetic_sim.stop()
            except Exception:
                pass

        # llamar constructores de simuladores, ver parametros si se necesita debuguear
        self.rotation_sim = RotationSimulation(debug=False)
        self.orbital_sim = OrbitalSimulation(self.rotation_sim)
        self.magnetic_sim = MagneticSimulation(self.orbital_sim, self.rotation_sim)

    def _start_simulators(self):
        """Inicializar hilos de cada simulador. Implementación paralelizada."""
        #TODO: Do not start the simulation thread, control them manually instead
        try:
            self.rotation_sim.start()
            self.orbital_sim.start()
            self.magnetic_sim.start()
        except Exception as e:
            print(f"Warning: Could not start all simulators: {e}")

    def _stop_simulators(self):
        """Detener los hilos de cada simulador."""
        try:
            if self.rotation_sim:
                self.rotation_sim.stop()
            if self.orbital_sim:
                self.orbital_sim.stop()
            if self.magnetic_sim:
                self.magnetic_sim.stop()
        except Exception as e:
            print(f"Warning: Error stopping simulators: {e}")

    def reset(self, seed=None, options=None):
        """
        Función para reiniciar el entorno y comenzar un nuevo episodio.
        Args:
            seed (int): Semilla aleatoria para reproducibilidad
            options (dict): Opciones adicionales (no utilizadas)

        Returns:
            tuple: (observación, información)
        """
        super().reset(seed=seed)

        # parar simulaciones para luego reiniciarlas para nuevo episodio
        self._stop_simulators()
        self._create_simulators()

        # condiciones iniciales aleatorias o fijas
        initial_angular_velocity = self.np_random.uniform(-1.0, 1.0, size=3)
        # initial_angular_velocity = self.np_random.uniform(-1.0, 1.0, size=3)
        # initial_angular_velocity = np.array([1.0, 0.0, 0.0])

        initial_quat = self.np_random.normal(size=4)
        # initial_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Setear condiciones iniciales
        initial_quat /= np.linalg.norm(initial_quat)
        self.rotation_sim.angular_velocity = initial_angular_velocity
        self.rotation_sim.quaternion = initial_quat

        # empezar simulaciones con nuevas condiciones
        self._start_simulators()

        # reiniciar tracking
        self.current_step = 0
        self.episode_reward = 0.0

        # esperar a que se reinicie todo, not the best solution pero funciona
        time.sleep(0.1)

        observation = self._get_observation()
        info = {}

        if self.render_mode == 'human':
            self.render()

        return observation, info

    def step(self, action):
        """
        Ejecuta un solo paso en el entorno dentro de un episodio.

        Args:
            action (np.ndarray): Comando en 3 dimensiones representando el torque

        Returns:
            tuple: (observación, recompensa, terminado, truncado, información) /
                   (observation, reward, terminated, truncated, info)
        """
        # mapear accion discreta a vector de torque
        torque_action = self.action_map[action]
        # print("torque action", torque_action)

        ### TEST: Compare with a simple proportional controller
        # G = 1e-3
        # torque_action = -self.rotation_sim.angular_velocity*G
        ###

        # 🔹 GUARDAR ESTADO ANTERIOR ANTES DE APLICAR ACCIÓN
        previous_angular_vel_norm = np.linalg.norm(self.rotation_sim.angular_velocity)

        # aplicar accion de torque al simulador de rotacion
        self.rotation_sim.set_torque(torque_action)

        # Avanzar la simulacion con una granularidad menor
        dt = self.time_step / self.sim_granularity
        for i in range(self.sim_granularity):
            self.current_time += timedelta(seconds=dt) # Avanzar el tiempo en el step definido
            self.rotation_sim.update_simulation(dt) # Actualizar la simulacion

            # obtener nueva observacion
            observation = self._get_observation()
            # Guardar historicos para graficar
            if self._debug:
                # Agregar el torque también al historico
                observation = np.concatenate((observation, torque_action))
                self._observation_hist.append(observation)
                # Agregar el tiempo al historico
                self._time_hist.append(self.current_time.timestamp())

        # 🔹 CALCULAR RECOMPENSA CON ESTADO ANTERIOR
        reward = self._calculate_reward(torque_action, previous_angular_vel_norm)
        self.episode_reward += reward

        # revisar si es que termino el episodio
        try:
            angular_vel_norm = np.linalg.norm(self.rotation_sim.angular_velocity)
            terminated = angular_vel_norm < self.success_threshold
        except Exception:
            angular_vel_norm = 1.0
            terminated = False

        # revisar si ocurre un timeout episodico
        # parametro "truncated" (revisar docs en gymnasium)
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        # retornar info adicional
        info = {
            'angular_velocity_norm': angular_vel_norm,
            'episode_reward': self.episode_reward,
            'success': terminated
        }

        if self.render_mode == 'human':
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """
        Obtener la observación actual del simulador.

        Returns:
            np.ndarray: Vector de observación: [quat(4), angular_vel(3), mag_field(3)]
        """
        try:
            # obtener estado del simulador de rotación
            quaternion = self.rotation_sim.quaternion.copy()
            angular_velocity = self.rotation_sim.angular_velocity.copy()

            # obtener info del campo magnetico
            try:
                mag_field_data = self.magnetic_sim.send_request('earth_magnetic_field').result()
                # extraer componentes x, y, z y convertir de nT a T
                mag_field_inertial = np.array([
                    mag_field_data['north'],
                    mag_field_data['east'],
                    mag_field_data['vertical']
                ]) * 1e-9

                # rotar campo magnetico de inercial a cuerpo usando quaternion
                mag_field_body = self._rotate_vector_by_quaternion(mag_field_inertial, quaternion)

            except Exception as e:
                print(f"Warning: Could not get magnetic field: {e}")
                mag_field_body = np.zeros(3)

            observation = np.concatenate([
                quaternion,
                angular_velocity,
                mag_field_body
            ]).astype(np.float32)

            return observation

        except Exception as e:
            print(f"Error getting observation: {e}")
            # retornar observacion default en caso de fallo
            return np.zeros(10, dtype=np.float32)

    def _rotate_vector_by_quaternion(self, vector, quaternion):
        """
        Rotar un vector del marco inercial al marco del cuerpo usando un quaternion.
        Wrapper de lo que ya existe en RotationSimulation.

        Args:
            vector (np.ndarray): Vector 3d en el marco inercial
            quaternion (np.ndarray): Quaternion [qx, qy, qz, qw]

        Returns:
            np.ndarray: Vector rotado en el marco del cuerpo
        """
        try:
            # representar vector como un quaternion puro
            v_quat = np.array([vector[0], vector[1], vector[2], 0.0])

            # conjugado del quaternion (inercial a cuerpo)
            q_conj = np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])

            # Usar el método estático existente para la multiplicación de quaterniones
            # rotated_v = q_conj * v_quat * q
            temp = RotationSimulation.quat_mut(q_conj, v_quat)
            rotated_v = RotationSimulation.quat_mut(temp, quaternion)

            # retornar solo la parte del vector
            return rotated_v[:3]

        except Exception as e:
            print(f"Warning: Quaternion rotation failed: {e}")
            return vector

    def _calculate_reward(self, action, previous_angular_vel_norm):
        """
        Función de recompensa simplificada con reward shaping.
        
        Args:
            action: Acción aplicada (torque)
            previous_angular_vel_norm: Norma de velocidad angular del paso anterior
        """
        try:
            current_angular_vel = self.rotation_sim.angular_velocity
            angular_vel_norm = np.linalg.norm(current_angular_vel)
        except Exception:
            angular_vel_norm = 1.0
        
        control_effort = np.linalg.norm(action)
        
        # Recompensa base: exponencial negativa (más sensible a cambios pequeños)
        base_reward = -np.exp(angular_vel_norm) + 1.0
        
        # Reward shaping: premiar mejora gradual
        improvement = previous_angular_vel_norm - angular_vel_norm
        shaped_reward = 10.0 * improvement  # Multiplicador alto para cambios pequeños
        
        # Penalización de control reducida
        control_penalty = -0.01 * control_effort
        
        # Bonus por lograr objetivo
        success_bonus = 20.0 if angular_vel_norm < self.success_threshold else 0.0
        
        reward = base_reward + shaped_reward + control_penalty + success_bonus
        
        return reward

    def render(self):
        """
        Renderizar estado actual del entorno.
        """
        if self.render_mode == 'human':
            try:
                quaternion = self.rotation_sim.quaternion
                angular_velocity = self.rotation_sim.angular_velocity
                angular_vel_norm = np.linalg.norm(angular_velocity)

                # print(f"Step: {self.current_step:3d} | "
                #       f"Time: {self.current_time.timestamp():.4f} | "
                #       f"ω_norm: {angular_vel_norm:.6f} rad/s | "
                #       f"ω: [{angular_velocity[0]:.6f}, {angular_velocity[1]:.6f}, {angular_velocity[2]:.6f}] rad/s | "
                #       f"Episode Reward: {self.episode_reward:.2f} | "
                #       f"Quaternion: [{quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f}]")
            except Exception as e:
                print(f"Render error: {e}")

        if self.render_mode == 'plot':
            pass

    def close(self):
        """
        Limpiar entorno y reiniciar todos los simuladores externos.
        """
        if self._plot_hist:
            self.show_hist()
        self._stop_simulators()

    def show_hist(self):
        if len(self._observation_hist) == 0:
            print("No hay historial guardado")
            return

        observation_hist = np.array(self._observation_hist)
        quat_hist = observation_hist[:,0:4]
        vel_hist = observation_hist[:,4:7]
        mag_hist = observation_hist[:,7:10]
        torque_hist = observation_hist[:,10:13]

        figure, axes = plt.subplots(3, 1)
        plt.title("Rotation Simulation")
        axes[0].grid(True)
        axes[1].grid(True)
        plt.ion()
        plt.show(block=False)

        axes[0].clear()
        axes[0].plot(self._time_hist, np.array(vel_hist), "--.", label=["x", "y", "z"])
        axes[0].legend(loc="upper right")
        axes[0].set_ylabel('Velocity (rad/s)')
        axes[0].set_xlabel('Time')
        axes[0].grid(True)

        axes[1].clear()
        axes[1].plot(self._time_hist, np.array(quat_hist), "--.", label=["i", "j", "k", "s"])
        axes[1].legend(loc="upper right")
        axes[1].set_ylabel('Quaternion')
        axes[1].set_xlabel('Time')
        axes[1].grid(True)

        axes[2].clear()
        axes[2].plot(self._time_hist, np.array(torque_hist), "--.", label=["Tx", "Ty", "Tz"])
        axes[2].legend(loc="upper right")
        axes[2].set_ylabel('Torque (%)')
        axes[2].set_xlabel('Time')
        axes[2].grid(True)

        plt.show(block=True)


def test_environment_basic():
    """
    Función de prueba para mostrar el funcionamiento del entorno.
    """
    print("=" * 60)
    print("Testing CubeSat Detumbling Environment")
    print("=" * 60)

    env = CubeSatDetumblingEnv(render_mode='human')

    try:
        # reiniciar entorno
        obs, _ = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print(f"Initial observation: {obs}")

        # tomar 10 acciones aleatorias
        for i in range(10):
            action = env.action_space.sample()
            print(f"\nStep {i + 1}: Action = {action}")

            obs, reward, terminated, truncated, _ = env.step(action)
            print(f"Reward: {reward:.4f}")

            if terminated or truncated:
                print(f"Episode ended at step {i + 1}")
                if terminated:
                    print("SUCCESS: Detumbling achieved!")
                else:
                    print("Episode truncated (timeout)")
                break

    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("Environment test completed!")

def evaluate_random_agent(episodes=100, max_steps=400):
    """
    Evalúa el desempeño de un agente aleatorio (línea base) y registra el historial
    de recompensas y éxitos por episodio.

    Args:
        episodes (int): Número de episodios para la evaluación.
        max_steps (int): Máximo de pasos por episodio.

    Returns:
        tuple: (métricas_finales, historial_recompensas, historial_exito)
    """
    print("=" * 70)
    print(f"🧪 EVALUACIÓN DEL AGENTE ALEATORIO (LÍNEA BASE) - {episodes} EPISODIOS")
    print("=" * 70)

    # Crear una instancia del entorno sin renderizado para la evaluación
    env = CubeSatDetumblingEnv(render_mode=None, max_steps=max_steps, debug=False, plot_hist=False)
    
    total_rewards = []
    success_count = 0
    steps_to_success = []
    
    # Listas para almacenar el historial de cada episodio
    episode_rewards_hist = []
    episode_success_hist = [] 
    
    start_time = time.time()

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            # 1. Selección de Acción: Aleatoria (probabilidad uniforme)
            action = env.action_space.sample()
            
            # 2. Paso en el entorno
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            done = terminated or truncated

        # 3. Registro de métricas y historial
        total_rewards.append(total_reward)
        episode_rewards_hist.append(total_reward)
        
        is_success = terminated
        episode_success_hist.append(is_success)
        
        if terminated:
            success_count += 1
            steps_to_success.append(step_count)
            
        if (episode + 1) % 10 == 0 or episode == episodes - 1:
            print(f"  Episodio {episode + 1}/{episodes} | Recompensa: {total_reward:.2f} | Éxito: {'SÍ' if terminated else 'NO'}")

    end_time = time.time()
    
    # 4. Cálculo de Métricas Finales
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    success_rate = success_count / episodes
    
    if success_count > 0:
        avg_steps_success = np.mean(steps_to_success)
        std_steps_success = np.std(steps_to_success)
    else:
        avg_steps_success = np.nan
        std_steps_success = np.nan

    env.close()

    metrics = {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'success_rate': success_rate,
        'avg_steps_on_success': avg_steps_success,
        'std_steps_on_success': std_steps_success,
        'total_time_s': end_time - start_time
    }

    print("\n" + "=" * 70)
    print("📊 RESUMEN DE MÉTRICAS (LÍNEA BASE ALEATORIA)")
    print("-" * 70)
    print(f"Recompensa Promedio (μ): **{metrics['mean_reward']:.2f}**")
    print(f"Desv. Estándar (σ):       {metrics['std_reward']:.2f}")
    print(f"Tasa de Éxito:            **{metrics['success_rate'] * 100:.2f}%**")
    print(f"Pasos Prom. al Éxito:     {metrics['avg_steps_on_success']:.1f} (solo si hay éxitos)")
    print("-" * 70)
    print(f"Tiempo total de simulación: {metrics['total_time_s']:.2f} segundos")
    print("=" * 70)

    # Retornar las métricas finales y los historiales
    return metrics, episode_rewards_hist, episode_success_hist

def plot_random_agent_history(episode_rewards, episode_success, window_size=10):
    """
    Genera gráficos de la ejecución del agente aleatorio.

    Args:
        episode_rewards (list): Lista de recompensas acumuladas por episodio.
        episode_success (list): Lista de booleanos indicando éxito por episodio.
        window_size (int): Tamaño de la ventana para la media móvil.
    """
    episodes = np.arange(1, len(episode_rewards) + 1)
    
    # 1. Suavizar la Recompensa (Media Móvil)
    rewards_series = np.array(episode_rewards)
    # Media móvil simple
    smoothed_rewards = np.convolve(rewards_series, np.ones(window_size)/window_size, mode='valid')
    # Ajustar el eje X para la media móvil
    episodes_smoothed = np.arange(window_size, len(episode_rewards) + 1)

    # 2. Calcular la Tasa de Éxito Acumulada
    success_rate_cumulative = np.cumsum(episode_success) / episodes

    # 3. Graficar
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
    fig.suptitle('Métricas del Agente Aleatorio (Línea Base)', fontsize=16, fontweight='bold')

    # --- Gráfico 1: Recompensa por Episodio (Suavizada) ---
    axes[0].plot(episodes_smoothed, smoothed_rewards, label=f'Media Móvil ({window_size} eps)', color='orange')
    axes[0].set_ylabel('Recompensa Acumulada (Media Móvil)')
    axes[0].set_title('Progreso de Recompensa por Episodio')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    # --- Gráfico 2: Tasa de Éxito Acumulada ---
    axes[1].plot(episodes, success_rate_cumulative, label='Tasa de Éxito Acumulada', color='green')
    axes[1].set_ylabel('Tasa de Éxito Acumulada')
    axes[1].set_xlabel('Episodio')
    axes[1].set_ylim(0, 1.05)
    axes[1].set_yticks(np.arange(0, 1.1, 0.1))
    
    # Línea de la tasa de éxito final
    if success_rate_cumulative.size > 0:
        final_rate = success_rate_cumulative[-1]
        axes[1].axhline(y=final_rate, color='r', linestyle=':', label=f'Tasa Final: {final_rate:.2f}')
    
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=True)

    print("\nGráficos de la Línea Base generados.")

if __name__ == "__main__":
    """
    Prueba simple en este mismo script.
    En caso de entrenamiento, se recomienda usar el script train_cubesat_detumbling.py.
    """
    print("=" * 60)
    print("CubeSat Detumbling Environment Test")
    print("=" * 60)

    debug = True  # Activar o desactivar gráficos
    plot_hist = True
    start_time = datetime.fromtimestamp(1758566834)
    time_step = 1
    total_time = 15*60
    granularity = 10

    # crear y probar el entorno
    env = CubeSatDetumblingEnv(render_mode='human', start_time=start_time, time_step=time_step, granularity=granularity,
                               debug=debug, plot_hist=plot_hist)

    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    """
    try:
        # correr un episodio de prueba...
        obs, _ = env.reset()
        print(f"\nInitial observation shape: {obs.shape}")
        print("Running N random steps...")

        for step in np.arange(0, total_time, time_step):
            action = env.action_space.sample()
            print(f"Acción: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Reward: {reward:.4f}")

            if terminated:
                print(f"\nSUCCESS! Episode completed at step {step + 1}")
                break
            elif truncated:
                print(f"\nEpisode truncated at step {step + 1}")
                break

    except Exception as e:
        print(f"Test error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()
        print("\nEnvironment test completed!")
    

    """
    # --- Configuración para la Línea Base ---
    EPISODES_TO_EVALUATE = 100 
    MAX_STEPS_PER_EPISODE = 400 # Asegúrate de que coincida con la configuración por defecto
    
    print("=" * 60)
    print("CubeSat Detumbling Environment Test")
    print("=" * 60)
    
    try:
        # 1. Ejecutar el Agente Aleatorio y obtener métricas E HISTORIAL
        # Capturamos los 3 valores devueltos: métricas finales, historial de recompensas e historial de éxito.
        random_agent_metrics, rewards_hist, success_hist = evaluate_random_agent(
            episodes=EPISODES_TO_EVALUATE, 
            max_steps=MAX_STEPS_PER_EPISODE
        )
        
        # 2. Generar gráficos del historial
        # Usamos los historiales capturados en el paso 1.
        plot_random_agent_history(rewards_hist, success_hist, window_size=10)
        
        # 3. Imprimir los resultados
        print("\n🎉 Evaluación de Línea Base completada.")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()

    # Cierra la instancia inicial del entorno si no se usó en el bloque try/except.
    # Si usaste el bloque de prueba comentado anteriormente, esto cerraría su instancia.
    try:
        env.close()
    except Exception:
        pass
    
    # Nota: Se recomienda mantener esta sección de prueba separada del código 
    # de entrenamiento real para una estructura más limpia.
    