from time import time

import numpy as np

from HopfieldNetwork import HopfieldNetwork
from image_processing import numpy_array_to_bmp, transform_images

NOISE_PERCENTAGES = [5,10,15,20, 30, 40, 50, 60, 70, 80, 90, 95]

def compare_images(expected, retrieved, time_start, time_end):
    """
    Compara el patrón esperado con el obtenido.
    Imprime si son idénticos y el porcentaje de píxeles/valores en los que varían.
    
    Parameters:
    expected  : array-like o np.ndarray original
    retrieved : array-like o np.ndarray devuelto por la red
    """
    exp = np.array(expected)
    ret = np.array(retrieved)
    
    total = len(exp)
    diff_count = np.sum(exp != ret)
    diff_percentage = (diff_count / total) * 100
    is_equal = diff_count == 0
    
    print(f"Recalling image ended in {(time_end - time_start):.2f} seconds | Diferencias: {diff_count} de {total} valores ({diff_percentage:.2f}%)")  
    
    return is_equal, diff_percentage

def add_noise(vector, noise_percentage):
    """
    Agrega ruido a un vector de valores (-1, 1) invirtiendo el signo de un porcentaje de sus elementos.
    
    Parameters:
    vector           : array-like o np.ndarray original (con valores -1 y 1)
    noise_percentage : float entre 0 y 100 (ej. 10 para 10% de ruido)
    
    Returns:
    np.ndarray       : copia del vector original con ruido aplicado
    """
    noisy_vector = np.array(vector).copy()
    total_elements = len(noisy_vector)
    
    # Cantidad de elementos a invertir según el porcentaje
    num_to_flip = int(total_elements * (noise_percentage / 100))
    
    # Seleccionamos índices aleatorios sin repetición
    flip_indices = np.random.choice(total_elements, size=num_to_flip, replace=False)
    
    # Invertimos el signo (1 pasa a -1, -1 pasa a 1)
    noisy_vector[flip_indices] *= -1
    
    return noisy_vector

def recall_all_images(Xs, noise_percetage, hopfield):
    i = 0
    for x in Xs:
        xn = add_noise(x, noise_percetage)
        # compare_images(x, xn)
        time_start = time()
        x = hopfield.recall(xn)
        time_end = time()
        # numpy_array_to_bmp(x, f"Imagen {i}.bmp")
        compare_images(xn, x, time_start, time_end)
        i += 1
def ej1():
    Xs = transform_images()
    hopfield = HopfieldNetwork(Xs)
    hopfield.train()
    for noise_percentage in NOISE_PERCENTAGES:
        print()
        print()
        print(f"Recalling images with {noise_percentage}% of noise")
        recall_all_images(Xs, noise_percentage, hopfield)