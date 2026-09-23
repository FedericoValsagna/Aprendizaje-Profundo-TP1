import os
import numpy as np

from HopfieldNetwork import HopfieldNetwork
from helper import add_noise, compare_images, delete_last_percentage
from image_processing import numpy_array_to_bmp, transform_images, image_name_from_id
from plots import plot_error_vs_borrado, plot_error_vs_noise

NOISE_PERCENTAGES = range(0, 82, 2)
OUTPUT_FOLDER = "./OutputEj1"
OUTPUT_FOLDER_JPG = "./OutputEj1Jpg"
OUTPUT_FOLDER_BORRADO_JPG = "./OutputEj1BorradoJpg"
REPETITIONS = 20


def recall_pattern(
    Xs: list,
    noise_percetage: float,
    hopfield: HopfieldNetwork,
    save_images: bool = False,
    verbose: bool = True,
):
    """
    Recupera cada patrón de Xs luego de agregarle el ruido indicado.
    Devuelve una lista con el porcentaje de error obtenido para cada imagen
    (mismo orden que Xs), para poder trackear cómo evoluciona el error por imagen.
    """
    error_percentages = []
    for i, x in enumerate(Xs):
        xn = add_noise(x, noise_percetage)
        new_x = hopfield.recall(xn)
        is_equal, diff_percentage, diff_count, total = compare_images(x, new_x)
        if verbose:
            print(
                f"Recalling image | Diferencias: {diff_count} de {total} valores ({diff_percentage:.2f}%)"
            )

        error_percentages.append(diff_percentage)

        if save_images:
            save_image(xn, new_x, i, noise_percetage)

    return error_percentages


def save_image(xn, new_x, i, noise_percetage):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    noisy_path = os.path.join(
        OUTPUT_FOLDER, f"img{i}_noise{noise_percetage}_ruidosa.bmp"
    )
    recalled_path = os.path.join(
        OUTPUT_FOLDER, f"img{i}_noise{noise_percetage}_recuperada.bmp"
    )
    numpy_array_to_bmp(xn, noisy_path)
    numpy_array_to_bmp(new_x, recalled_path)

    os.makedirs(OUTPUT_FOLDER_JPG, exist_ok=True)
    noisy_path_jpg = os.path.join(
        OUTPUT_FOLDER_JPG, f"img{i}_noise{noise_percetage}_ruidosa.jpg"
    )
    recalled_path_jpg = os.path.join(
        OUTPUT_FOLDER_JPG, f"img{i}_noise{noise_percetage}_recuperada.jpg"
    )
    numpy_array_to_bmp(xn, noisy_path_jpg)
    numpy_array_to_bmp(new_x, recalled_path_jpg)


DELETE_PERCENTAGES = [20, 30, 50]
OUTPUT_FOLDER_BORRADO = "./OutputEj1Borrado"


def recall_pattern_deleted(
    Xs: list,
    delete_percentage: float,
    hopfield: HopfieldNetwork,
    save_images: bool = False,
    verbose: bool = True,
):
    error_percentages = []
    for i, x in enumerate(Xs):
        xd = delete_last_percentage(x, delete_percentage)
        new_x = hopfield.recall(xd)
        is_equal, diff_percentage, diff_count, total = compare_images(x, new_x)
        if verbose:
            print(
                f"Recalling image (borrado {delete_percentage}%) | Diferencias: {diff_count} de {total} valores ({diff_percentage:.2f}%)"
            )

        error_percentages.append(diff_percentage)

        if save_images:
            save_image_deleted(xd, new_x, i, delete_percentage)

    return error_percentages


def save_image_deleted(xd, new_x, i, delete_percentage):
    os.makedirs(OUTPUT_FOLDER_BORRADO, exist_ok=True)
    deleted_path = os.path.join(
        OUTPUT_FOLDER_BORRADO, f"img{i}_borrado{delete_percentage}_incompleta.bmp"
    )
    recalled_path = os.path.join(
        OUTPUT_FOLDER_BORRADO, f"img{i}_borrado{delete_percentage}_recuperada.bmp"
    )
    numpy_array_to_bmp(xd, deleted_path)
    numpy_array_to_bmp(new_x, recalled_path)

    os.makedirs(OUTPUT_FOLDER_BORRADO_JPG, exist_ok=True)
    deleted_path_jpg = os.path.join(
        OUTPUT_FOLDER_BORRADO_JPG, f"img{i}_borrado{delete_percentage}_incompleta.jpg"
    )
    recalled_path_jpg = os.path.join(
        OUTPUT_FOLDER_BORRADO_JPG, f"img{i}_borrado{delete_percentage}_recuperada.jpg"
    )
    numpy_array_to_bmp(xd, deleted_path_jpg)
    numpy_array_to_bmp(new_x, recalled_path_jpg)


def ej1_borrado():
    print("Ejercicio 1 - patrones con sección borrada")
    Xs = transform_images()
    hopfield = HopfieldNetwork(Xs)
    hopfield.train()

    results_by_image = {i: [] for i in range(len(Xs))}

    for delete_percentage in DELETE_PERCENTAGES:
        print()
        print(
            f"Recalling images with {delete_percentage}% borrado ({REPETITIONS} corridas)"
        )

        errors_per_image = [[] for _ in range(len(Xs))]

        for rep in range(REPETITIONS):
            is_first_run = rep == 0
            error_percentages = recall_pattern_deleted(
                Xs,
                delete_percentage,
                hopfield,
                save_images=is_first_run,
                verbose=is_first_run,
            )
            for i, error_percentage in enumerate(error_percentages):
                errors_per_image[i].append(error_percentage)

        for i in range(len(Xs)):
            mean_error = float(np.mean(errors_per_image[i]))
            results_by_image[i].append((delete_percentage, mean_error))

        print("All Images recalled. See errors above")

    plot_error_vs_borrado(results_by_image)


def ej1_ruido():
    Xs = transform_images()
    hopfield = HopfieldNetwork(Xs)
    hopfield.train()

    results_by_image = {i: [] for i in range(len(Xs))}

    for noise_percentage in NOISE_PERCENTAGES:
        print()
        print()
        print(
            f"Recalling images with {noise_percentage}% of noise ({REPETITIONS} corridas)"
        )

        errors_per_image = [[] for _ in range(len(Xs))]

        for rep in range(REPETITIONS):
            is_first_run = rep == 0
            error_percentages = recall_pattern(
                Xs,
                noise_percentage,
                hopfield,
                save_images=is_first_run,
                verbose=is_first_run,
            )
            for i, error_percentage in enumerate(error_percentages):
                errors_per_image[i].append(error_percentage)

        for i in range(len(Xs)):
            mean_error = float(np.mean(errors_per_image[i]))
            results_by_image[i].append((noise_percentage, mean_error))

        print("All Images recalled. See errors above")

    plot_error_vs_noise(results_by_image)


def ej1():
    print("Ejercicio 1")
    # ej1_ruido()
    ej1_borrado()
