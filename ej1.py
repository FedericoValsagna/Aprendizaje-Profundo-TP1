import os
from time import time

import numpy as np

import Hopfield2
from HopfieldNetwork import HopfieldNetwork
from image_processing import numpy_array_to_bmp, transform_images

NOISE_PERCENTAGES = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 35, 37, 38, 39, 40, 50, 60, 70]
OUTPUT_FOLDER = "./OutputEj1"

def compare_images(expected, retrieved):
    exp = np.array(expected)
    ret = np.array(retrieved)
    
    total = len(exp)
    diff_count = np.sum(exp != ret)
    diff_percentage = (diff_count / total) * 100
    is_equal = diff_count == 0
    
    return is_equal, diff_percentage, diff_count, total

def add_noise(vector: list, noise_percentage: int):
    noisy_vector = np.array(vector).copy()
    total_elements = len(noisy_vector)
    
    num_to_flip = int(total_elements * (noise_percentage / 100))
    flip_indices = np.random.choice(total_elements, size=num_to_flip, replace=False)
    noisy_vector[flip_indices] *= -1
    
    return noisy_vector

def recall_pattern(Xs: list, noise_percetage: float, hopfield: HopfieldNetwork, is_image: bool = False):
    for i, x in enumerate(Xs):
        xn = add_noise(x, noise_percetage)
        time_start = time()
        new_x = hopfield.recall(xn)
        time_end = time()
        is_equal, diff_percentage, diff_count, total = compare_images(x, new_x)
        print(f"Recalling image ended in {(time_end - time_start):.2f} seconds | Diferencias: {diff_count} de {total} valores ({diff_percentage:.2f}%)")
        
        if is_image:
            is_image(xn, new_x, i, noise_percetage)
        
def save_image(xn, new_x, i, noise_percetage):
    noisy_path = os.path.join(OUTPUT_FOLDER, f"img{i}_noise{noise_percetage}_ruidosa.bmp")
    recalled_path = os.path.join(OUTPUT_FOLDER, f"img{i}_noise{noise_percetage}_recuperada.bmp")
    numpy_array_to_bmp(xn, noisy_path)
    numpy_array_to_bmp(new_x, recalled_path)
    
    
def ej1():
    print("Ejercicio 1")
    Xs = transform_images()
    hopfield = HopfieldNetwork(Xs)
    hopfield.train()
    for noise_percentage in NOISE_PERCENTAGES:
        print()
        print()
        print(f"Recalling images with {noise_percentage}% of noise")
        recall_pattern(Xs, noise_percentage, hopfield, is_image=True)
        print("All Images recalled. See errors above")
