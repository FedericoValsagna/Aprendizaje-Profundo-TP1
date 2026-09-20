from PIL import Image
import numpy as np
import os
FOLDERNAME = "Imagenes"


def transformación(matrix):
    matrix = matrix.astype(np.int8)
    matrix = matrix.flatten()
    # print(matrix)
    for i in range(len(matrix)):
        if matrix[i] >= 0:
            matrix[i] = 1
        else:
            matrix[i] = -1
    return matrix



def transform_images():
    Xs = []
    for file in imagenes():
        # print("\n--- Procesando:", file)
        matriz = bmp_to_numpy_array(os.path.join(FOLDERNAME, file))
        # print("Array de NumPy a partir de la matriz:")
        # print(transformación(matriz))
        # numpy_array_to_bmp(matriz, os.path.join(FOLDERNAME, f"output_{file}"))
        Xs.append(transformación(matriz))
    return Xs


def bmp_to_numpy_array(file_path):
    imagen = Image.open(file_path)
    matriz = np.array(imagen)
    
    # Si la imagen es booleana (monocromática), convertimos True->255 y False->0
    if matriz.dtype == bool:
        matriz = matriz.astype(np.uint8) * 255
    else:
        matriz = matriz.astype(np.uint8)

    # print(matriz)
    return matriz

def numpy_array_to_bmp(matriz, file_path):
    imagen = Image.fromarray(matriz)
    imagen.save(file_path)

def imagenes():
    return [file for file in os.listdir(FOLDERNAME) if file.lower().endswith(".bmp")]