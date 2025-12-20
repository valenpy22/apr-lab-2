import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def load_qtable(filepath="q_table4.pkl"):
    """Cargar Q-table y metadata."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


def print_metadata(data):
    """Imprimir información sobre la Q-table."""
    print("=" * 70)
    print("📊 Q-TABLE METADATA")
    print("=" * 70)
    
    print(f"\n🔢 Dimensiones:")
    # print(f"   • Número de episodios: {data['total_episodes']:,}")
    print(f"   • Número de estados: {data['num_states']:,}")
    print(f"   • Número de acciones: {data['num_actions']}")
    print(f"   • Bins por eje: {data['n_bins']}")
    print(f"   • Shape Q-table: {data['q_table'].shape}")
    print(f"   • Tamaño en memoria: {data['q_table'].nbytes / 1024 / 1024:.2f} MB")
    
    print(f"\n⚙️ Hiperparámetros:")
    for key, value in data['hyperparameters'].items():
        print(f"   • {key}: {value}")
    
    print(f"\n📈 Estadísticas de Q-values:")
    q_table = data['q_table']
    print(f"   • Mínimo: {np.min(q_table):.4f}")
    print(f"   • Máximo: {np.max(q_table):.4f}")
    print(f"   • Media: {np.mean(q_table):.4f}")
    print(f"   • Desv. estándar: {np.std(q_table):.4f}")
    print(f"   • Valores no-cero: {np.count_nonzero(q_table):,} ({100*np.count_nonzero(q_table)/q_table.size:.2f}%)")
    
    print(f"\n🎯 Bins de velocidad angular:")
    for i, bins in enumerate(data['ang_vel_bins']):
        axis_name = ['X', 'Y', 'Z'][i]
        print(f"   • Eje {axis_name}: [{bins[0]:.3f}, {bins[-1]:.3f}] rad/s")
        print(f"     Primeros 5: {bins[:5]}")
        print(f"     Últimos 5: {bins[-5:]}")
    
    print("=" * 70)


def visualize_qtable_heatmap(data, max_states=100):
    """Visualizar Q-table como heatmap (primeros N estados)."""
    q_table = data['q_table']
    
    # Limitar número de estados para visualización
    states_to_show = min(max_states, q_table.shape[0])
    
    plt.figure(figsize=(14, 8))
    
    # Heatmap de Q-values
    plt.subplot(2, 1, 1)
    sns.heatmap(q_table[:states_to_show, :], 
                cmap='RdYlGn', 
                center=0,
                cbar_kws={'label': 'Q-value'},
                xticklabels=True,
                yticklabels=False)
    plt.title(f'Q-table Heatmap (primeros {states_to_show} estados)')
    plt.xlabel('Acción')
    plt.ylabel('Estado')
    
    # Distribución de Q-values
    plt.subplot(2, 1, 2)
    plt.hist(q_table.flatten(), bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribución de Q-values')
    plt.xlabel('Q-value')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qtable_heatmap.png', dpi=150, bbox_inches='tight')
    print("💾 Guardado: qtable_heatmap.png")
    plt.show()


def visualize_best_actions_by_velocity(data):
    """Visualizar mejores acciones para diferentes velocidades angulares."""
    q_table = data['q_table']
    bins = data['ang_vel_bins']
    n_bins = data['n_bins']
    
    # Crear grillas de velocidades
    # Simplificamos a 2D: solo X e Y, fijando Z=0
    z_center_idx = n_bins // 2  # Índice del centro (velocidad Z ≈ 0)
    
    best_actions = np.zeros((n_bins, n_bins))
    max_q_values = np.zeros((n_bins, n_bins))
    
    for i in range(n_bins):
        for j in range(n_bins):
            # Estado = [i, j, z_center_idx]
            state_idx = i * (n_bins ** 2) + j * n_bins + z_center_idx
            if state_idx < q_table.shape[0]:
                best_action = np.argmax(q_table[state_idx, :])
                best_actions[i, j] = best_action
                max_q_values[i, j] = np.max(q_table[state_idx, :])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Mejores acciones
    im1 = axes[0].imshow(best_actions, cmap='tab20', origin='lower', aspect='auto')
    axes[0].set_title('Mejor Acción por Estado (Z ≈ 0)')
    axes[0].set_xlabel('Velocidad Angular Y (bins)')
    axes[0].set_ylabel('Velocidad Angular X (bins)')
    
    # Agregar colorbar con acciones
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('ID de Acción')
    
    # Agregar etiquetas de velocidad
    x_ticks = np.linspace(0, n_bins-1, 5, dtype=int)
    y_ticks = np.linspace(0, n_bins-1, 5, dtype=int)
    axes[0].set_xticks(x_ticks)
    axes[0].set_yticks(y_ticks)
    axes[0].set_xticklabels([f'{bins[1][i]:.2f}' for i in x_ticks])
    axes[0].set_yticklabels([f'{bins[0][i]:.2f}' for i in y_ticks])
    
    # Máximo Q-value
    im2 = axes[1].imshow(max_q_values, cmap='viridis', origin='lower', aspect='auto')
    axes[1].set_title('Máximo Q-value por Estado (Z ≈ 0)')
    axes[1].set_xlabel('Velocidad Angular Y (bins)')
    axes[1].set_ylabel('Velocidad Angular X (bins)')
    
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Max Q-value')
    
    axes[1].set_xticks(x_ticks)
    axes[1].set_yticks(y_ticks)
    axes[1].set_xticklabels([f'{bins[1][i]:.2f}' for i in x_ticks])
    axes[1].set_yticklabels([f'{bins[0][i]:.2f}' for i in y_ticks])
    
    plt.tight_layout()
    plt.savefig('qtable_policy_map.png', dpi=150, bbox_inches='tight')
    print("💾 Guardado: qtable_policy_map.png")
    plt.show()


def visualize_action_distribution(data):
    """Visualizar distribución de acciones óptimas."""
    q_table = data['q_table']
    
    # Obtener mejor acción para cada estado
    best_actions = np.argmax(q_table, axis=1)
    
    plt.figure(figsize=(12, 6))
    
    # Histograma de acciones
    plt.subplot(1, 2, 1)
    action_counts = np.bincount(best_actions, minlength=data['num_actions'])
    plt.bar(range(data['num_actions']), action_counts, edgecolor='black', alpha=0.7)
    plt.title('Distribución de Acciones Óptimas')
    plt.xlabel('ID de Acción')
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Porcentaje
    plt.subplot(1, 2, 2)
    action_pcts = 100 * action_counts / np.sum(action_counts)
    colors = plt.cm.viridis(np.linspace(0, 1, len(action_pcts)))
    plt.barh(range(data['num_actions']), action_pcts, color=colors, edgecolor='black')
    plt.title('Porcentaje de Uso por Acción')
    plt.xlabel('Porcentaje (%)')
    plt.ylabel('ID de Acción')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Agregar valores en las barras
    for i, (count, pct) in enumerate(zip(action_counts, action_pcts)):
        plt.text(pct + 0.5, i, f'{pct:.1f}%', va='center')
    
    plt.tight_layout()
    plt.savefig('qtable_action_distribution.png', dpi=150, bbox_inches='tight')
    print("💾 Guardado: qtable_action_distribution.png")
    plt.show()


def visualize_learning_coverage(data):
    """Visualizar qué estados han sido visitados durante el entrenamiento."""
    q_table = data['q_table']
    
    # Estados visitados = estados con al menos un Q-value no-cero
    visited_states = np.any(q_table != 0, axis=1)
    n_visited = np.sum(visited_states)
    coverage = 100 * n_visited / len(visited_states)
    
    plt.figure(figsize=(14, 6))
    
    # Mapa de cobertura
    plt.subplot(1, 2, 1)
    plt.imshow(visited_states.reshape(-1, 1).T, cmap='RdYlGn', aspect='auto')
    plt.title(f'Estados Visitados: {n_visited}/{len(visited_states)} ({coverage:.1f}%)')
    plt.xlabel('Estado')
    plt.ylabel('Visitado')
    plt.yticks([])
    
    # Por bloques
    plt.subplot(1, 2, 2)
    block_size = 100
    n_blocks = len(visited_states) // block_size + 1
    coverage_per_block = []
    
    for i in range(n_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, len(visited_states))
        block_coverage = 100 * np.sum(visited_states[start:end]) / (end - start)
        coverage_per_block.append(block_coverage)
    
    plt.bar(range(n_blocks), coverage_per_block, edgecolor='black', alpha=0.7)
    plt.title(f'Cobertura por Bloques de {block_size} Estados')
    plt.xlabel('Bloque')
    plt.ylabel('Cobertura (%)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=coverage, color='r', linestyle='--', label=f'Media: {coverage:.1f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('qtable_coverage.png', dpi=150, bbox_inches='tight')
    print("💾 Guardado: qtable_coverage.png")
    plt.show()
    
    print(f"\n📊 Cobertura del espacio de estados: {coverage:.2f}%")
    print(f"   • Estados visitados: {n_visited:,}")
    print(f"   • Estados no visitados: {len(visited_states) - n_visited:,}")


def visualize_q_values_by_state_distance(data):
    """Visualizar Q-values en función de la distancia al objetivo."""
    q_table = data['q_table']
    bins = data['ang_vel_bins']
    n_bins = data['n_bins']
    
    # Calcular distancia al objetivo (velocidad angular = 0) para cada estado
    distances = []
    max_q_values = []
    
    for i in range(n_bins):
        for j in range(n_bins):
            for k in range(n_bins):
                state_idx = i * (n_bins ** 2) + j * n_bins + k
                if state_idx < q_table.shape[0]:
                    # Velocidad angular correspondiente
                    vel_x = bins[0][i]
                    vel_y = bins[1][j]
                    vel_z = bins[2][k]
                    
                    # Distancia euclideana al origen
                    distance = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
                    distances.append(distance)
                    max_q_values.append(np.max(q_table[state_idx, :]))
    
    distances = np.array(distances)
    max_q_values = np.array(max_q_values)
    
    # Filtrar valores no-cero (estados visitados)
    mask = max_q_values != 0
    distances_visited = distances[mask]
    max_q_visited = max_q_values[mask]
    
    plt.figure(figsize=(12, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(distances_visited, max_q_visited, alpha=0.3, s=10)
    plt.title('Q-value vs Distancia al Objetivo')
    plt.xlabel('||ω|| (rad/s)')
    plt.ylabel('Max Q-value')
    plt.grid(True, alpha=0.3)
    
    # Binned average
    plt.subplot(1, 2, 2)
    n_bins_plot = 20
    dist_bins = np.linspace(0, np.max(distances_visited), n_bins_plot)
    bin_indices = np.digitize(distances_visited, dist_bins)
    
    bin_means = []
    bin_centers = []
    for i in range(1, n_bins_plot):
        mask = bin_indices == i
        if np.any(mask):
            bin_means.append(np.mean(max_q_visited[mask]))
            bin_centers.append((dist_bins[i-1] + dist_bins[i]) / 2)
    
    plt.plot(bin_centers, bin_means, 'o-', linewidth=2, markersize=8)
    plt.title('Q-value Promedio vs Distancia')
    plt.xlabel('||ω|| (rad/s)')
    plt.ylabel('Q-value Promedio')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qtable_distance_analysis.png', dpi=150, bbox_inches='tight')
    print("💾 Guardado: qtable_distance_analysis.png")
    plt.show()


def generate_full_report(filepath="q_table4.pkl"):
    """Generar reporte completo de visualización."""
    print("\n🚀 Generando reporte completo de Q-table...\n")
    
    # Cargar datos
    data = load_qtable(filepath)
    
    # 1. Metadata
    print_metadata(data)
    
    print("\n📊 Generando visualizaciones...\n")
    
    # 2. Heatmap y distribución
    print("1️⃣ Heatmap y distribución de Q-values...")
    visualize_qtable_heatmap(data, max_states=100)
    
    # 3. Mapa de política
    print("\n2️⃣ Mapa de política (mejores acciones)...")
    visualize_best_actions_by_velocity(data)
    
    # 4. Distribución de acciones
    print("\n3️⃣ Distribución de acciones...")
    visualize_action_distribution(data)
    
    # 5. Cobertura de exploración
    print("\n4️⃣ Cobertura de exploración...")
    visualize_learning_coverage(data)
    
    # 6. Análisis por distancia
    # print("\n5️⃣ Análisis por distancia al objetivo...")
    # visualize_q_values_by_state_distance(data)
    
    print("\n" + "=" * 70)
    print("✅ Reporte completo generado!")
    print("=" * 70)
    print("\nArchivos guardados:")
    print("  • qtable_heatmap.png")
    print("  • qtable_policy_map.png")
    print("  • qtable_action_distribution.png")
    print("  • qtable_coverage.png")
    print("  • qtable_distance_analysis.png")

def analyze_unvisited_states(data):
    """Analizar qué velocidades angulares corresponden a estados no visitados."""
    q_table = data['q_table']
    bins = data['ang_vel_bins']
    n_bins = len(bins[0])
    
    # Estados visitados
    visited = np.any(q_table != 0, axis=1)
    
    # Analizar primeros 300 estados
    print("Análisis de primeros 300 estados:")
    print("=" * 60)
    
    velocities_unvisited = []
    velocities_visited = []
    
    for state in range(min(300, len(visited))):
        i = state // (n_bins ** 2)
        j = (state % (n_bins ** 2)) // n_bins
        k = state % n_bins
        
        vel = np.array([bins[0][i], bins[1][j], bins[2][k]])
        
        if visited[state]:
            velocities_visited.append(vel)
        else:
            velocities_unvisited.append(vel)
    
    if velocities_unvisited:
        velocities_unvisited = np.array(velocities_unvisited)
        print(f"\n❌ Estados NO visitados (primeros 300): {len(velocities_unvisited)}")
        print(f"   Rango ωx: [{velocities_unvisited[:, 0].min():.3f}, {velocities_unvisited[:, 0].max():.3f}]")
        print(f"   Rango ωy: [{velocities_unvisited[:, 1].min():.3f}, {velocities_unvisited[:, 1].max():.3f}]")
        print(f"   Rango ωz: [{velocities_unvisited[:, 2].min():.3f}, {velocities_unvisited[:, 2].max():.3f}]")
    
    if velocities_visited:
        velocities_visited = np.array(velocities_visited)
        print(f"\n✅ Estados VISITADOS (primeros 300): {len(velocities_visited)}")
        print(f"   Rango ωx: [{velocities_visited[:, 0].min():.3f}, {velocities_visited[:, 0].max():.3f}]")
        print(f"   Rango ωy: [{velocities_visited[:, 1].min():.3f}, {velocities_visited[:, 1].max():.3f}]")
        print(f"   Rango ωz: [{velocities_visited[:, 2].min():.3f}, {velocities_visited[:, 2].max():.3f}]")


if __name__ == "__main__":
    # Generar reporte completo
    # generate_full_report("q_table_final.pkl")

    # Ejecutar
    # data = load_qtable("q_table5.pkl")
    # analyze_unvisited_states(data)
    
    # O usar funciones individuales:
    data = load_qtable("q_table5_best.pkl")
    print_metadata(data)
    # visualize_qtable_heatmap(data)
    # visualize_best_actions_by_velocity(data)