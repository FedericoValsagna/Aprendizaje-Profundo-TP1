import numpy as np


def add_noise(vector: list, noise_percentage: int):
    noisy_vector = np.array(vector).copy()
    total_elements = len(noisy_vector)
    
    num_to_flip = int(total_elements * (noise_percentage / 100))
    flip_indices = np.random.choice(total_elements, size=num_to_flip, replace=False)
    noisy_vector[flip_indices] *= -1
    
    return noisy_vector


def compare_images(expected, retrieved):
    exp = np.array(expected)
    ret = np.array(retrieved)
    
    total = len(exp)
    diff_count = np.sum(exp != ret)
    diff_percentage = (diff_count / total) * 100
    is_equal = diff_count == 0
    
    return is_equal, diff_percentage, diff_count, total


def delete_last_percentage(vector: list, delete_percentage: float, fill_value: int = -1):
    """
    Borra (reemplaza por fill_value) el último delete_percentage% de los elementos
    del vector, simulando una imagen con una porción faltante.
    """
    deleted_vector = np.array(vector).copy()
    total_elements = len(deleted_vector)

    num_to_delete = int(total_elements * (delete_percentage / 100))
    if num_to_delete > 0:
        deleted_vector[-num_to_delete:] = fill_value

    return deleted_vector