import numpy as np
from HopfieldNetwork import HopfieldNetwork
from plots import (
    plot_perror_vs_pN_correlaciones,
    plot_perror_vs_pN_tabla,
)

N = 1000
REPETITIONS = 10
OUTPUT_FOLDER = "./OutputEj2"
TABLA_PERROR = [0.001, 0.0036, 0.01, 0.05, 0.1]
TABLA_PMAXOVERN = [0.105, 0.138, 0.185, 0.37, 0.61]


CORRELACIONES = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3]
PERROR_OBJETIVO = 0.01


def generar_patrones(largo, n):
    return [np.random.choice([1, -1], size=largo) for _ in range(n)]


def perror_sincronico(hopfield: HopfieldNetwork, xs: list):
    errores = []
    for x in xs:
        x = np.array(x, dtype=float)
        x_new = np.sign(hopfield.W @ x)
        x_new[x_new == 0] = 1
        error = np.mean(x_new != x)
        errores.append(error)
    return np.mean(errores)


def medir_perror_vs_p(N, valores_p, repeticiones=REPETITIONS):
    resultados = []
    for p in valores_p:
        errores_rep = []
        for _ in range(repeticiones):
            xs = generar_patrones(N, p)
            hopfield = HopfieldNetwork(xs)
            hopfield.train()
            errores_rep.append(perror_sincronico(hopfield, xs))
        resultados.append((p, np.mean(errores_rep)))
        print(f"p={p}, p/N={p/N:.3f} -> Perror empírico = {np.mean(errores_rep):.5f}")
    return resultados


def ej2a():
    valores_p = list(range(50, 650, 25))
    resultados = medir_perror_vs_p(N, valores_p)
    plot_perror_vs_pN_tabla(
        resultados, N, pmax_sobre_n=TABLA_PMAXOVERN, tabla_perror=TABLA_PERROR
    )


def generar_patrones_correlacionados(largo, n, rho):
    q = (1 - np.sqrt(rho)) / 2
    base = np.random.choice([1, -1], size=largo)
    patrones = []
    for _ in range(n):
        flips = np.where(np.random.rand(largo) < q, -1, 1)
        patrones.append(base * flips)
    return patrones


def correlacion_media(xs):
    X = np.array(xs, dtype=float)
    C = (X @ X.T) / X.shape[1]
    p = len(xs)
    return (C.sum() - np.trace(C)) / (p * (p - 1))


def perror_promedio(N, p, rho, repeticiones=REPETITIONS):
    errores = []
    for _ in range(repeticiones):
        xs = generar_patrones_correlacionados(N, p, rho)
        hopfield = HopfieldNetwork(xs)
        hopfield.train()
        errores.append(perror_sincronico(hopfield, xs))
    return np.mean(errores)


def calcular_pmax(N, rho, perror_objetivo=PERROR_OBJETIVO, paso=5, p_max_busqueda=800):
    p_ok, err_ok = 2, 0.0
    for p in range(paso, p_max_busqueda + 1, paso):
        err = perror_promedio(N, p, rho)
        if err > perror_objetivo:
            frac = (perror_objetivo - err_ok) / (err - err_ok)
            return p_ok + frac * (p - p_ok)
        p_ok, err_ok = p, err
    return p_max_busqueda


def medir_perror_vs_p_correlacion(N, valores_p, rho, repeticiones=REPETITIONS):
    resultados = []
    for p in valores_p:
        err = perror_promedio(N, p, rho, repeticiones)
        resultados.append((p, err))
        print(f"rho={rho:.2f}, p={p}, p/N={p/N:.3f} -> Perror = {err:.5f}")
    return resultados


def ej2b():
    valores_p = list(range(25, 650, 25))
    resultados_por_rho = {}
    for rho in CORRELACIONES:
        rho_emp = correlacion_media(generar_patrones_correlacionados(N, 50, rho))
        print(f"--- rho={rho:.2f} (correlación empírica {rho_emp:.3f}) ---")
        resultados_por_rho[rho] = medir_perror_vs_p_correlacion(N, valores_p, rho)
    plot_perror_vs_pN_correlaciones(resultados_por_rho, N)


def ej2b_capacidad():
    resultados = []
    for rho in CORRELACIONES:
        pmax = calcular_pmax(N, rho)
        resultados.append((rho, pmax))
        print(f"rho={rho:.2f} -> pmax={pmax:.1f}, pmax/N={pmax/N:.3f}")


def ej2():
    ej2a()
    ej2b()
