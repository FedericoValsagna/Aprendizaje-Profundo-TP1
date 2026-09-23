import numpy as np
from HopfieldNetwork import HopfieldNetwork
from plots import plot_perror_vs_pN_tabla

N = 1000
REPETITIONS = 10
OUTPUT_FOLDER = "./OutputEj2"
TABLA_PERROR = [0.001, 0.0036, 0.01, 0.05, 0.1]
TABLA_PMAXOVERN = [0.105, 0.138, 0.185, 0.37, 0.61]

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


    
def ej2():
    valores_p = list(range(50, 650, 25))
    resultados = medir_perror_vs_p(N, valores_p)
    plot_perror_vs_pN_tabla(resultados, N, pmax_sobre_n=TABLA_PMAXOVERN, tabla_perror=TABLA_PERROR)
     
