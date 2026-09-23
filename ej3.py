import numpy as np
from HopfieldNetwork import HopfieldNetwork
from ej2 import generar_patrones, TABLA_PERROR
from plots import plot_capacidad_vs_poda, plot_energia_vs_pasos, plot_error_vs_ruido_por_poda
from helper import add_noise, compare_images

N = 1000
REPETITIONS = 10
OUTPUT_FOLDER = "./OutputEj3"
REPETICIONES_RUIDO = 3


def podar_neuronas(W: np.ndarray, porcentaje: float) -> np.ndarray:
    W_podada = W.copy()
    l = W.shape[0]

    i_upper, j_upper = np.triu_indices(l, k=1)
    total_conexiones = len(i_upper)
    num_a_eliminar = int(total_conexiones * (porcentaje / 100))

    if num_a_eliminar == 0:
        return W_podada

    idx = np.random.choice(total_conexiones, size=num_a_eliminar, replace=False)
    W_podada[i_upper[idx], j_upper[idx]] = 0
    W_podada[j_upper[idx], i_upper[idx]] = 0

    return W_podada


def evaluar_ruido_vs_poda(N: int, p: int, porcentajes_poda: list, noise_percentages: list,
                           repeticiones: int = REPETICIONES_RUIDO):

    xs = generar_patrones(N, p)
    hopfield = HopfieldNetwork(xs)
    hopfield.train()
    W_original = hopfield.W.copy()

    resultados_por_poda = {}
    for poda_pct in porcentajes_poda:
        hopfield.W = podar_neuronas(W_original, poda_pct)
        puntos = []
        for noise_pct in noise_percentages:
            errores = []
            for _ in range(repeticiones):
                for x in xs:
                    xn = add_noise(x, noise_pct)
                    recuperado = hopfield.recall(xn)
                    _, diff_pct, _, _ = compare_images(x, recuperado)
                    errores.append(diff_pct)
            error_promedio = float(np.mean(errores))
            puntos.append((noise_pct, error_promedio))
            print(f"p={p} poda={poda_pct}% ruido={noise_pct}% -> error promedio={error_promedio:.2f}%")
        resultados_por_poda[poda_pct] = puntos
    return resultados_por_poda





def perror_sincronico_con_W(W: np.ndarray, xs: list) -> float:
    errores = []
    for x in xs:
        x = np.array(x, dtype=float)
        x_new = np.sign(W @ x)
        x_new[x_new == 0] = 1
        error = np.mean(x_new != x)
        errores.append(error)
    return np.mean(errores)


def see_error_evolution(N: int, valores_p: list, porcentaje_poda: float,
                                repeticiones: int = REPETITIONS):
    resultados = []
    for p in valores_p:
        errores_rep = []
        for _ in range(repeticiones):
            xs = generar_patrones(N, p)
            hopfield = HopfieldNetwork(xs)
            hopfield.train()
            W_podada = podar_neuronas(hopfield.W, porcentaje_poda)
            errores_rep.append(perror_sincronico_con_W(W_podada, xs))
        resultados.append((p, np.mean(errores_rep)))
    return resultados


def calculate_pmax(resultados, perror_target):
    p_vals = np.array([r[0] for r in resultados])
    perror_vals = np.array([r[1] for r in resultados])

    order = np.argsort(perror_vals)
    perror_sorted = perror_vals[order]
    p_sorted = p_vals[order]

    if perror_target < perror_sorted.min() or perror_target > perror_sorted.max():
        return np.nan
    return np.interp(perror_target, perror_sorted, p_sorted)


def see_capacity_evolution(N: int, valores_p: list, porcentajes_poda: list,
                             perror_target: float = 0.01, repeticiones: int = REPETITIONS):
    resultados_capacidad = []
    for porcentaje in porcentajes_poda:
        print(f"\nMidiendo capacidad con {porcentaje}% de sinapsis eliminadas...")
        resultados_p = see_error_evolution(N, valores_p, porcentaje, repeticiones)
        pmax = calculate_pmax(resultados_p, perror_target)
        pmax_sobre_n = pmax / N if not np.isnan(pmax) else np.nan
        resultados_capacidad.append((porcentaje, pmax_sobre_n))
        print(f"poda={porcentaje}% -> pmax/N (Perror={perror_target}) = {pmax_sobre_n:.4f}")
    return resultados_capacidad





def ej3_1():
    n = 1000
    p = 100
    poda_levels_ruido = [0, 20, 40, 60, 80]
    noise_percentages = range(2,50,2)
    
    print(f"\n--- Error vs. ruido por poda, P grande (N={n}, p={p}) ---")
    resultados_grande = evaluar_ruido_vs_poda(n, p, poda_levels_ruido, noise_percentages)
    plot_error_vs_ruido_por_poda(resultados_grande, p, N)

def ej3_2():
    valores_p = list(range(50, 400, 25))
    perror_target = 0.01
    porcentajes_poda_capacidad = [0, 10, 20, 30, 40, 50, 60]
    print(f"Capacidad vs. poda (N={N})")
    resultados_capacidad = see_capacity_evolution(
        N, valores_p, porcentajes_poda_capacidad, perror_target=perror_target
    )
    plot_capacidad_vs_poda(resultados_capacidad, N, perror_target)


def ej3_3():
    n = 1000
    p = 50
    noise_percentage = 20
    porcentajes_poda = [0, 20, 50, 80]
    
    
    print(f"Energía durante el recall asincrónico (N={N})")
    xs = generar_patrones(n, p)
    hopfield = HopfieldNetwork(xs)
    hopfield.train()
    patron_objetivo = xs[0]
    xs = add_noise(patron_objetivo, noise_percentage)

    curvas = {}
    for porcentaje in porcentajes_poda:
        W_podada = podar_neuronas(hopfield.W, porcentaje)
        _, energias = hopfield.recall_con_energia(W_podada, xs.copy())
        etiqueta = "Red completa" if porcentaje == 0 else f"{porcentaje}% eliminado"
        curvas[etiqueta] = energias
        print(f"{etiqueta}: energía inicial={energias[0]:.2f}, final={energias[-1]:.2f}, pasos={len(energias) - 1}")

    plot_energia_vs_pasos(curvas)


def ej3():
    print("Ejercicio 3")
    # ej3_1()
    ej3_2()
    # ej3_3()