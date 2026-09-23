import os
from matplotlib import pyplot as plt
import numpy as np
from image_processing import image_name_from_id

OUTPUT_FOLDER_EJ1 = "./OutputEj1"
OUTPUT_FOLDER_EJ2 = "./OutputEj2"
OUTPUT_FOLDER_EJ3 = "./OutputEj3"
OUTPUT_FOLDER_EJ1_BORRADO = "./OutputEj1Borrado"

def plot_energia_vs_pasos(curvas: dict, output_path=None):
    if output_path is None:
        output_path = os.path.join(OUTPUT_FOLDER_EJ3, "energia_vs_pasos.png")
    os.makedirs(OUTPUT_FOLDER_EJ3, exist_ok=True)
    plt.figure(figsize=(9, 6))
    for etiqueta, energias in curvas.items():
        plt.plot(energias, label=etiqueta)

    plt.xlabel("Número de actualizaciones (pasos asincrónicos)")
    plt.ylabel("Energía")
    plt.title("Evolución de la energía durante el recall asincrónico")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en {output_path}")
    
    
def plot_capacidad_vs_poda(resultados_capacidad, N, perror_target, output_path=None):
    if output_path is None:
        output_path = os.path.join(OUTPUT_FOLDER_EJ3, "capacidad_vs_poda.png")
    os.makedirs(OUTPUT_FOLDER_EJ3, exist_ok=True)
    porcentajes = [r[0] for r in resultados_capacidad]
    pmax_sobre_n = [r[1] for r in resultados_capacidad]

    plt.figure(figsize=(9, 6))
    plt.plot(porcentajes, pmax_sobre_n, "o-", color="tab:green")
    plt.xlabel("Porcentaje de sinapsis eliminadas (%)")
    plt.ylabel("pmax / N")
    plt.title(f"Capacidad vs. porcentaje de sinapsis eliminadas\n(N={N}, Perror objetivo={perror_target})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en {output_path}")


def plot_error_vs_ruido_por_poda(resultados_por_poda: dict, p: int, N: int, output_path=None):
    if output_path is None:
        output_path = os.path.join(OUTPUT_FOLDER_EJ3, f"error_vs_ruido_p{p}.png")
    os.makedirs(OUTPUT_FOLDER_EJ3, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for poda_pct, puntos in sorted(resultados_por_poda.items()):
        puntos_ordenados = sorted(puntos, key=lambda t: t[0])
        noise_vals = [t[0] for t in puntos_ordenados]
        error_vals = [t[1] for t in puntos_ordenados]
        etiqueta = "Red completa" if poda_pct == 0 else f"{poda_pct}% sinapsis eliminadas"
        plt.plot(noise_vals, error_vals, "o-", label=etiqueta)

    plt.xlabel("Porcentaje de ruido (%)")
    plt.ylabel("Porcentaje de error en el recall (%)")
    plt.title(f"Error de recall vs. ruido, por nivel de poda (N={N}, p={p})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en {output_path}")
    
def plot_error_vs_noise(results_by_image: dict, output_path: str = None):
    if output_path is None:
        output_path = os.path.join(OUTPUT_FOLDER_EJ1, "error_vs_noise.png")

    os.makedirs(OUTPUT_FOLDER_EJ1, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for image_index, points in sorted(results_by_image.items()):
        points_sorted = sorted(points, key=lambda p: p[0])
        noise_values = [p[0] for p in points_sorted]
        mean_values = [p[1] for p in points_sorted]
        plt.plot(noise_values, mean_values, label=image_name_from_id(image_index))

    plt.xlabel("Porcentaje de ruido (%)")
    plt.ylabel("Porcentaje de error en el recall (%)")
    plt.title(f"Evolución del error de recall según el ruido, por imagen\n(promedio de {OUTPUT_FOLDER_EJ1} corridas por nivel de ruido)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en {output_path}")
    
def plot_error_vs_borrado(results_by_image: dict, output_path: str = None):
    if output_path is None:
        output_path = os.path.join(OUTPUT_FOLDER_EJ1_BORRADO, "error_vs_borrado.png")

    os.makedirs(OUTPUT_FOLDER_EJ1_BORRADO, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for image_index, points in sorted(results_by_image.items()):
        points_sorted = sorted(points, key=lambda p: p[0])
        borrado_values = [p[0] for p in points_sorted]
        mean_values = [p[1] for p in points_sorted]
        plt.plot(borrado_values, mean_values, "o-", label=image_name_from_id(image_index))

    plt.xlabel("Porcentaje de la imagen borrada (%)")
    plt.ylabel("Porcentaje de error en el recall (%)")
    plt.title("Evolución del error de recall según la porción borrada, por imagen")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en {output_path}")


def plot_perror_vs_pN_tabla(resultados, N, pmax_sobre_n, tabla_perror):
    output_path = os.path.join(OUTPUT_FOLDER_EJ2, "perror_vs_pN_tabla.png")
    os.makedirs(OUTPUT_FOLDER_EJ2, exist_ok=True)

    p_vals = np.array([r[0] for r in resultados])
    perror_emp = np.array([r[1] for r in resultados])
    pN_emp = p_vals / N

    plt.figure(figsize=(9, 6))
    plt.plot(pN_emp, perror_emp, "o-", label="Perror empírico", color="tab:blue")
    plt.plot(pmax_sobre_n, tabla_perror, "s--", label="Tabla 2.1 (Hertz et al.)", color="tab:orange")

    plt.xlabel("p/N")
    plt.ylabel("Perror")
    plt.yscale("log")
    plt.title(f"Perror vs p/N: empírico vs Tabla 2.1 (N={N})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en {output_path}")
    
    
    
def plot_perror_vs_pN_correlaciones(resultados_por_rho, N):
    os.makedirs(OUTPUT_FOLDER_EJ2, exist_ok=True)
    plt.figure(figsize=(9, 6))
    for rho, resultados in sorted(resultados_por_rho.items()):
        pN = np.array([r[0] for r in resultados]) / N
        perror = np.array([r[1] for r in resultados])
        mask = perror > 0  # la escala log no admite Perror = 0
        plt.plot(pN[mask], perror[mask], "o-", markersize=3, label=f"ρ = {rho:.1f}")
 
    plt.xlabel("p/N")
    plt.ylabel("Perror")
    plt.yscale("log")
    plt.title(f"Perror vs p/N para distintos grados de correlación (N={N})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_FOLDER_EJ2, "perror_vs_pN_correlaciones.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en {path}")